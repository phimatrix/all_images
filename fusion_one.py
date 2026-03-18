import os
import ee
import time
import traceback

os.environ['HTTP_PROXY'] = 'http://rrsceast:NRSC%40User@192.168.0.9:8080'
os.environ['HTTPS_PROXY'] = 'http://rrsceast:NRSC%40User@192.168.0.9:8080'

PROJECT_ID = "theta-arcana-484116-h7"
START_DATE = '2023-07-01'
END_DATE = '2023-07-31'
OPTICAL_MISSION = 'S2'
PCA_SMOOTH = True
PCA_COMPONENT_RATIO = 0.1
STD_CLOUD_THRESHOLD = 30

ee.Initialize(project=PROJECT_ID)

# ------------------------------------------------------------------
# Helper functions (same as before, with minor improvements)
# ------------------------------------------------------------------
def prepare_optical(collection, aoi, mission):
    def calc_ndvi(img):
        if mission == 'S2':
            nir, red = 'B8', 'B4'
        else:
            nir, red = 'B5', 'B4'
        ndvi = img.normalizedDifference([nir, red]).rename('NDVI')
        return img.addBands(ndvi)

    def mask_s2(img):
        scl = img.select('SCL')
        mask = scl.eq(3).Or(scl.gte(8)).rename('Mask')
        return img.addBands(mask)

    collection = collection.map(calc_ndvi).map(mask_s2)
    return collection.select(['NDVI', 'Mask'])

def cal_covariates(img, aoi, indep_vars):
    img = img.updateMask(img.gt(-40))
    spatial_mean = img.select(['VV','VH']).reduceRegion(
        ee.Reducer.mean(), aoi, 100, maxPixels=1e9).toImage(['VV','VH'])
    neighbor_mean = img.select(['VV','VH']).reduceNeighborhood(
        ee.Reducer.mean(), ee.Kernel.square(10))
    spatial_diff = img.select(['VV','VH']).subtract(spatial_mean)
    neighbor_diff = img.select(['VV','VH']).subtract(neighbor_mean.rename('VV_mean','VH_mean'))
    img = img.addBands(spatial_mean.rename('VV_mean','VH_mean')) \
             .addBands(neighbor_mean.rename('VV_Nmean','VH_Nmean')) \
             .addBands(spatial_diff.rename('VV_diff','VH_diff')) \
             .addBands(neighbor_diff.rename('VV_Ndiff','VH_Ndiff'))
    return img.select(indep_vars).focal_median().toFloat()

def pair_opt_sar(optical, sar, aoi, indep_vars):
    def pair_image(img):
        date = ee.Date(img.get('system:time_start'))
        sar_filtered = sar.filterDate(date.advance(-12, 'day'), date.advance(12, 'day'))
        sar_comp = sar_filtered.mean()
        covariates = cal_covariates(sar_comp, aoi, indep_vars)
        img = ee.Algorithms.If(sar_filtered.size().gt(0),
                               img.addBands(covariates),
                               img)
        return ee.Image(img).set('S1_COUNT', sar_filtered.size())
    paired = optical.map(pair_image).filterMetadata('S1_COUNT', 'greater_than', 0)
    return paired

def export_image(img, aoi, name, folder):
    try:
        task = ee.batch.Export.image.toDrive(
            image=img,
            description=name,
            fileNamePrefix=name,
            folder=folder,
            scale=10,
            region=aoi,
            maxPixels=1e13
        )
        task.start()
        print(f"  ✅ Started: {name}")
        return task
    except Exception as e:
        print(f"  ❌ Failed to start {name}: {e}")
        return None

# ------------------------------------------------------------------
# Main processing function
# ------------------------------------------------------------------
def process_grid(grid_number):
    grid_id = f"grid_{grid_number}"
    print(f"\n{'='*60}")
    print(f"PROCESSING {grid_id}")
    print('='*60)

    try:
        aoi = ee.FeatureCollection(f"projects/{PROJECT_ID}/assets/{grid_id}").geometry()
    except Exception as e:
        print(f"❌ Cannot access AOI: {e}")
        return

    # Load data
    print("\n📡 Loading Sentinel-2...")
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(aoi).filterDate(START_DATE, END_DATE)
    s2_count = s2.size().getInfo()
    print(f"   → {s2_count} images")

    print("🛰️ Loading Sentinel-1...")
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(aoi).filterDate(START_DATE, END_DATE) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    s1_count = s1.size().getInfo()
    print(f"   → {s1_count} images")

    if s2_count == 0 or s1_count == 0:
        print("❌ Insufficient data, skipping.")
        return

    # 1. True Color
    print("\n🖼️ Creating True Color...")
    s2_median = s2.median().clip(aoi)
    true_color = s2_median.select(['B4','B3','B2'])

    # 2. Raw NDVI
    print("📊 Creating Raw NDVI...")
    ndvi_raw = s2_median.normalizedDifference(['B8','B4']).rename('NDVI_raw')

    # 3 & 4. SAR VV and VH
    print("📡 Extracting SAR bands...")
    s1_mean = s1.mean().clip(aoi)
    vv = s1_mean.select('VV').rename('VV')
    vh = s1_mean.select('VH').rename('VH')

    # 5. Cloud Masked NDVI
    print("☁️ Creating Cloud‑masked NDVI...")
    optical_prep = prepare_optical(s2, aoi, OPTICAL_MISSION)
    cloud_masked_ndvi = optical_prep.select('NDVI').mean().clamp(-1,1).clip(aoi)

    # 6. Reconstructed NDVI – try complex method first
    print("🔧 Attempting accurate reconstruction...")
    indep_vars = ['VV_mean','VH_mean','VV_diff','VH_diff',
                  'VV_Nmean','VH_Nmean','VV_Ndiff','VH_Ndiff']
    paired = pair_opt_sar(optical_prep, s1, aoi, indep_vars)
    pair_count = paired.size().getInfo()
    print(f"   → {pair_count} image pairs found")

    reconstructed = None
    if pair_count >= 3:
        try:
            paired = paired.map(lambda img: img.addBands(
                img.select(0).multiply(0).add(1).rename('constant')))
            train = paired.select(['constant'] + indep_vars + ['NDVI'])
            regression = train.reduce(ee.Reducer.robustLinearRegression(9,1))
            coeffs = regression.select('coefficients').arrayFlatten(
                [['constant']+indep_vars, ['NDVI']])

            def predict(img):
                pred = img.select(['constant']+indep_vars).multiply(
                    coeffs.rename(['constant']+indep_vars)).reduce('sum').rename('NDVI_pred')
                return img.addBands(pred)

            outputs = paired.map(predict)

            if PCA_SMOOTH:
                n = ee.Number(pair_count).multiply(PCA_COMPONENT_RATIO).floor().int()
                smoothed = temporal_pca(outputs.select('NDVI_pred'), aoi, n, 50)
            else:
                smoothed = outputs.select('NDVI_pred').toBands()

            calibrated, filled = post_process(outputs, smoothed, aoi, STD_CLOUD_THRESHOLD)
            # Mean across all paired images → single band
            reconstructed = filled.select('NDVI').mean().clip(aoi)
            print("   ✅ Complex method succeeded")
        except Exception as e:
            print(f"   ⚠️ Complex method failed: {e}")
            reconstructed = None

    # Fallback: simple linear regression using VV and VH only
    if reconstructed is None:
        print("   ↪ Using fallback simple regression")
        try:
            training = vv.addBands(vh).addBands(cloud_masked_ndvi) \
                .sample(region=aoi, scale=10, numPixels=2000)
            regression = training.reduceColumns(
                reducer=ee.Reducer.linearRegression(2,1),
                selectors=['VV','VH','NDVI']
            )
            coeffs = ee.Array(regression.get('coefficients'))
            reconstructed = vv.expression(
                'a * VV + b * VH + c', {
                    'VV': vv, 'VH': vh,
                    'a': coeffs.get([0,0]),
                    'b': coeffs.get([1,0]),
                    'c': coeffs.get([2,0])
                }).clamp(-1,1).rename('NDVI_reconstructed').clip(aoi)
            print("   ✅ Fallback succeeded")
        except Exception as e:
            print(f"   ❌ Fallback also failed: {e}")
            return

    # Final clamp and rename (single band guaranteed)
    reconstructed = reconstructed.clamp(-1,1).rename('NDVI_reconstructed')

    # Export all six images
    folder = f"ISRO_Accurate_{grid_id}"
    print(f"\n📤 Exporting to Google Drive folder: {folder}")
    export_image(true_color, aoi, f"1_TrueColor_{grid_id}", folder)
    time.sleep(2)
    export_image(ndvi_raw, aoi, f"2_Raw_NDVI_{grid_id}", folder)
    time.sleep(2)
    export_image(vv, aoi, f"3_S1_VV_{grid_id}", folder)
    time.sleep(2)
    export_image(vh, aoi, f"4_S1_VH_{grid_id}", folder)
    time.sleep(2)
    export_image(cloud_masked_ndvi, aoi, f"5_CloudMasked_NDVI_{grid_id}", folder)
    time.sleep(2)
    export_image(reconstructed, aoi, f"6_Reconstructed_NDVI_{grid_id}", folder)

    print(f"\n✅ All exports attempted for {grid_id}. Check Google Drive and GEE Task Manager.")
    print("   Monitor at: https://code.earthengine.google.com/tasks")

# ------------------------------------------------------------------
# Missing functions that were not defined earlier
# ------------------------------------------------------------------
def temporal_pca(image_collection, aoi, num_components, scale=10):
    image = image_collection.toBands().clip(aoi)
    band_names = image.bandNames()
    mean_dict = image.reduceRegion(ee.Reducer.mean(), aoi, scale, maxPixels=1e9)
    means = ee.Image.constant(mean_dict.values(band_names))
    centered = image.subtract(means)
    arrays = centered.toArray()
    covar = arrays.reduceRegion(ee.Reducer.centeredCovariance(), aoi, scale, maxPixels=1e9)
    covar_array = ee.Array(covar.get('array'))
    eigens = covar_array.eigen()
    eigen_vectors = eigens.slice(1,1)
    array_image = arrays.toArray(1)
    pcs = ee.Image(eigen_vectors).matrixMultiply(array_image)
    selected = pcs.arraySlice(0,0,num_components)
    inverse = ee.Image(eigen_vectors.transpose().slice(0,0,num_components)).matrixMultiply(selected)
    result = inverse.arrayProject([0]).arrayFlatten([band_names]).add(means)
    return result

def post_process(paired, prediction, aoi, cloud_thresh):
    def calibrate(img):
        img_id = img.id().cat('_NDVI_pred')
        mask = img.select('Mask')
        cloud_cover = ee.Number(img.get('CLOUD_PERCENTAGE_AOI'))
        ndvi_pred = prediction.select(img_id).rename('NDVI')
        ndvi_obs = img.select('NDVI')
        calibrated = ee.Image(ndvi_pred).where(mask.Not(), ndvi_obs)
        return img.addBands(calibrated.rename('NDVI_pred'), overwrite=True)

    calibrated = paired.map(calibrate)

    def fill_gaps(img):
        pred = img.select('NDVI_pred')
        obs = img.select('NDVI')
        mask = img.select('Mask')
        filled = obs.where(mask, pred).focal_median()
        return ee.Image(filled).rename('NDVI').copyProperties(img, ['system:time_start'])
    filled = calibrated.map(fill_gaps)
    return calibrated, filled

if __name__ == "__main__":
    grid_num = input("Enter grid number to process (e.g., 210): ")
    process_grid(grid_num)

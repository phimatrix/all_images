import ee
import os

os.environ['HTTP_PROXY'] = 'http://rrsceast:NRSC%40User@192.168.0.9:8080'
os.environ['HTTPS_PROXY'] = 'http://rrsceast:NRSC%40User@192.168.0.9:8080'

PROJECT_ID = "theta-arcana-484116-h7"
ee.Initialize(project=PROJECT_ID)
print("Earth Engine initialized successfully.")

# Test access to one AOI
aoi = ee.FeatureCollection(f"projects/{PROJECT_ID}/assets/grid_210")
print(f"AOI grid_210 size: {aoi.size().getInfo()}")

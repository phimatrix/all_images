import time
from fusion_one import process_grid  # reuse the function

for i in range(201, 251):
    print(f"\n--- Starting grid {i} ---")
    try:
        process_grid(str(i))
    except Exception as e:
        print(f"Error on grid {i}: {e}")
    # Wait between grids to avoid overwhelming the task queue
    time.sleep(30)

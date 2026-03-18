import ee
import os

# Set proxy if needed
os.environ['HTTP_PROXY'] = 'http://rrsceast:NRSC%40User@192.168.0.9:8080'
os.environ['HTTPS_PROXY'] = 'http://rrsceast:NRSC%40User@192.168.0.9:8080'

print("Authenticating Earth Engine...")
ee.Authenticate()
print("Authentication complete. You can now initialize.")

import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://climate-api.open-meteo.com/v1/climate"
params = {
	"latitude": 7.1475,
	"longitude": 3.3619,
	"start_date": "2015-01-01",
	"end_date": "2050-12-31",
	"models": ["CMCC_CM2_VHR4", "FGOALS_f3_H", "HiRAM_SIT_HR", "MRI_AGCM3_2_S", "EC_Earth3P_HR", "MPI_ESM1_2_XR", "NICAM16_8S"],
	"timezone": "Europe/Berlin",
	"daily": ["temperature_2m_max", "temperature_2m_mean", "temperature_2m_min", "wind_speed_10m_mean", "relative_humidity_2m_mean", "relative_humidity_2m_max", "rain_sum", "precipitation_sum", "pressure_msl_mean", "cloud_cover_mean"],
}
responses = openmeteo.weather_api(url, params=params)

# Process 1 location and 7 models
for response in responses:
	print(f"\nCoordinates: {response.Latitude()}°N {response.Longitude()}°E")
	print(f"Elevation: {response.Elevation()} m asl")
	print(f"Timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
	print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")
	print(f"Model Nº: {response.Model()}")
	
	# Process daily data. The order of variables needs to be the same as requested.
	daily = response.Daily()
	daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
	daily_temperature_2m_mean = daily.Variables(1).ValuesAsNumpy()
	daily_temperature_2m_min = daily.Variables(2).ValuesAsNumpy()
	daily_wind_speed_10m_mean = daily.Variables(3).ValuesAsNumpy()
	daily_relative_humidity_2m_mean = daily.Variables(4).ValuesAsNumpy()
	daily_relative_humidity_2m_max = daily.Variables(5).ValuesAsNumpy()
	daily_rain_sum = daily.Variables(6).ValuesAsNumpy()
	daily_precipitation_sum = daily.Variables(7).ValuesAsNumpy()
	daily_pressure_msl_mean = daily.Variables(8).ValuesAsNumpy()
	daily_cloud_cover_mean = daily.Variables(9).ValuesAsNumpy()
	
	daily_data = {"date": pd.date_range(
		start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
		end =  pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = daily.Interval()),
		inclusive = "left"
	)}
	
	daily_data["temperature_2m_max"] = daily_temperature_2m_max
	daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
	daily_data["temperature_2m_min"] = daily_temperature_2m_min
	daily_data["wind_speed_10m_mean"] = daily_wind_speed_10m_mean
	daily_data["relative_humidity_2m_mean"] = daily_relative_humidity_2m_mean
	daily_data["relative_humidity_2m_max"] = daily_relative_humidity_2m_max
	daily_data["rain_sum"] = daily_rain_sum
	daily_data["precipitation_sum"] = daily_precipitation_sum
	daily_data["pressure_msl_mean"] = daily_pressure_msl_mean
	daily_data["cloud_cover_mean"] = daily_cloud_cover_mean
	
	daily_dataframe = pd.DataFrame(data = daily_data)
	print("\nDaily data\n", daily_dataframe)
	# You can save the dataframe to a CSV file if needed
	daily_dataframe.to_csv(f"daily_data_model_{response.Model()}.csv", index=False)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def format_pop_density(file_name):
	df = pd.read_csv(file_name, delimiter='\t')
	df.to_csv(os.path.join("data", "pop_density.csv"), index=False)

def extract_state_data(dir_name):
	# Split shit up into their states
	dfs = []
	for file in os.listdir(dir_name):
		dfs.append(pd.read_csv(os.path.join(dir_name, file)))
	data = pd.concat(dfs, axis=0, ignore_index=True)

	# get relevant weather stations only
	name_group = data.groupby("NAME")
	for state, station in weather_stations.items():
		df = name_group.get_group(station)
		if df.shape[0] <320:
			print(state, df.shape)
		df.to_csv(os.path.join("rdata/raw_data/aw_weather_data", state + ".csv"), index=False)

def format_weather_data(file_name, chunk=2):
	data = pd.read_csv(file_name)
	state = os.path.splitext(os.path.basename(file_name))[0]

	# Trim the data down to relevant parameters
	rel_data = data.groupby("NAME").get_group(weather_stations[state])[["NAME","DATE","TMIN","TAVG","TMAX"]]

	# Clean the data if needed
	if rel_data.isnull().values.any():
		rel_data["TMIN"] = rel_data["TMIN"].fillna(rel_data["TAVG"])
		rel_data["TMAX"] = rel_data["TMAX"].fillna(rel_data["TAVG"])
		rel_data["TAVG"] = rel_data["TAVG"].fillna((rel_data["TMAX"] + rel_data["TMIN"])/2)

	# Write the trimmed data to a file
	rel_data_file_path = os.path.join("data/weather_data_daily", state + ".csv")
	rel_data.to_csv(rel_data_file_path, index=False)

	# Group Data in x week chuncks
	n = chunk * 7
	data_chunks = [rel_data[i:i+n] for i in range(0,rel_data.shape[0],n)]

	columns = ["NAME", "WEEK", "START_DATE", "STOP_DATE", "TMIN", "TAVG", "TMAX", "TSTD"]
	chunked_data = []
	for i in range(len(data_chunks)):
		data_chunk = data_chunks[i]
		chunk_stats = [weather_stations[state], i, data_chunk["DATE"].min(), data_chunk["DATE"].max(), 
			data_chunk["TMIN"].min(), data_chunk["TAVG"].mean(), data_chunk["TMAX"].max(), data_chunk["TAVG"].std(ddof=1)]
		chunked_data.append(chunk_stats)

	chunked_data = np.array(chunked_data).reshape(len(data_chunks), len(columns))
	chuncked_data_df = pd.DataFrame(chunked_data, columns=columns)

	chunk_dir = 'data/weather_data_' + str(chunk) + 'w_chunks'
	if not os.path.exists(chunk_dir):
		os.makedirs(chunk_dir)

	chuncked_data_df.to_csv(os.path.join(chunk_dir, state + ".csv"), index=False)

if __name__ == "__main__":	

	# # Consolidate demographic data
	base_data = pd.read_csv("base_data.csv")
	census_data = pd.read_csv("data/raw_data/county_demo_data.csv")[['fips', 'POP010210', 'AGE135212', 'AGE295212', 'AGE775212', 'PVY020211', 'POP060210']]
	census_data.columns = ['COUNTY_FIPS', '2010_POP', 'PERCENT_UNDER_5', 'PERCENT_UNDER_18', 'PERCENT_OVER_65', 'PERCENT_BELOW_POVERTY_LINE', 'POP_DENSITY']
	data = pd.merge(base_data, census_data, on='COUNTY_FIPS', how='inner')
	print(data)
	print(data.shape)
	data.to_csv("data/demographic_data.csv", index=False)

	base_data = pd.read_csv("base_data.csv")
	mask_data = pd.read_csv("data/raw_data/covid-19-data/mask-use/mask-use-by-county.csv")
	mask_data.columns = ['COUNTY_FIPS', 'WEARS_MASK_NEVER', 'WEARS_MASK_RARELY', 'WEARS_MASK_SOMETIMES', 'WEARS_MASK_FREQUENTLY', 'WEARS_MASK_ALWAYS']

	data = pd.merge(base_data, mask_data, on='COUNTY_FIPS', how='inner')
	data.insert(5, 'WEARS_MASKS_REGULARLY', data['WEARS_MASK_FREQUENTLY'] + data['WEARS_MASK_ALWAYS'])
	data.to_csv("data/relevant_mask_data.csv", index=False)

	base_data = pd.read_csv("base_data.csv")
	covid_case_data = pd.read_csv("data/raw_data/covid-19-data/us-counties.csv")
	covid_case_data.columns = ['DATE', 'COUNTY', 'STATE', 'COUNTY_FIPS', 'CASES', 'DEATHS']
	covid_data = pd.merge(base_data, covid_case_data[['DATE', 'COUNTY_FIPS', 'CASES']], on='COUNTY_FIPS', how='inner')
	
	covid_data['DATE'] = pd.to_datetime(covid_data['DATE'])
	new_df = covid_data.groupby('STATE_ID')['DATE', 'CASES'].transform('min')
	new_df.columns = ['FIRST_DATE', 'CASES']
	
	covid_data = pd.concat([covid_data, (covid_data['DATE'] - new_df['FIRST_DATE'])/np.timedelta64(1, 'D'), (covid_data['CASES'] - covid_data['CASES'].shift()), (covid_data['CASES'] - covid_data['CASES'].shift())/covid_data['CASES'].shift()], axis=1)
	covid_data.columns = ['STATE_ID','STATE','CITY','COUNTY_NAME','COUNTY_FIPS','DATE','CASES','NUM_DAYS_SINCE_FIRST_CASE','NEW_CASES','PER_INCREASE_IN_CASES']
	covid_data['NEW_CASES'] = covid_data['NEW_CASES'].fillna(0)
	covid_data['PER_INCREASE_IN_CASES'] = covid_data['PER_INCREASE_IN_CASES'].fillna(0)
	
	covid_data.to_csv("data/relevant_covid_case_data.csv", index=False)

	data = pd.DataFrame()
	for file in os.listdir('data/raw_data/raw_weather_data'):
		state_w_data = pd.read_csv("data/raw_data/raw_weather_data/" + file)
		state_w_data = state_w_data[['DATE', 'NAME', 'TMAX', 'TMIN', 'AWND']]
		state_w_data.columns = ['DATE', 'WEATHER_STATION', 'TMAX', 'TMIN', 'AVG_DAILY_WIND_SPEED']
		state_w_data['STATE_ID'] = [os.path.splitext(os.path.basename(file))[0]] * state_w_data.shape[0]
		data = data.append(state_w_data, ignore_index=True)

	data.to_csv("data/relevant_weather_data.csv", index=False)

	base_data = pd.read_csv("base_data.csv")
	daily_trip_data = pd.read_csv("data/raw_data/Trips_by_Distance.csv")[['Level', 'Date', 'County FIPS', 'Population Staying at Home', 'Population Not Staying at Home', 'Number of Trips']]
	daily_trip_data.columns = ['LVL', 'DATE', 'COUNTY_FIPS', 'POP_AT_HOME', 'POP_NOT_AT_HOME', 'NUM_OF_TRIPS']
	daily_trip_data['DATE'] = pd.to_datetime(daily_trip_data['DATE'])
	rel_daily_trip_data = daily_trip_data.loc[daily_trip_data['LVL'] == 'County']
	data = pd.merge(base_data, rel_daily_trip_data[['LVL', 'DATE', 'COUNTY_FIPS', 'POP_AT_HOME', 'POP_NOT_AT_HOME', 'NUM_OF_TRIPS']], on=['COUNTY_FIPS'], how='inner')
	data = data[data['DATE']>pd.Timestamp('2020-01-01')]
	data.to_csv("data/relevant_travel_data.csv", index=False)

	demo_data = pd.read_csv("data/demographic_data.csv")
	weather_data = pd.read_csv("data/relevant_weather_data.csv")
	covid_data = pd.read_csv("data/relevant_covid_case_data.csv")
	mask_data = pd.read_csv("data/relevant_mask_data.csv")
	travel_data = pd.read_csv("data/relevant_travel_data.csv")

	covid_and_weather = pd.merge(covid_data[['STATE_ID', 'STATE', 'COUNTY_NAME', 'COUNTY_FIPS', 'DATE',
       'CASES','NUM_DAYS_SINCE_FIRST_CASE','NEW_CASES','PER_INCREASE_IN_CASES']], weather_data[['STATE_ID', 
       'DATE', 'TMAX', 'TMIN', 'AVG_DAILY_WIND_SPEED']], on=['STATE_ID', 'DATE'], how='inner')

	covid_and_weather['DATE'] = pd.to_datetime(covid_and_weather['DATE'])
	weather_data['DATE'] = pd.to_datetime(weather_data['DATE'])

	two_week_tmax_avg = []
	two_week_tmin_avg = []
	two_week_wnd_speed_avg = []
	for index, row in covid_and_weather.iterrows():
 		two_week_wd = weather_data[(weather_data['DATE'] < row['DATE']) & (weather_data['DATE'] >= (row['DATE'] - pd.Timedelta(days=14))) & (weather_data['STATE_ID'] == row['STATE_ID'])]
 		two_week_tmax_avg.append(two_week_wd['TMAX'].mean())
 		two_week_tmin_avg.append(two_week_wd['TMIN'].mean())
 		two_week_wnd_speed_avg.append(two_week_wd['AVG_DAILY_WIND_SPEED'].mean())

	covid_and_weather['2W_TMAX_AVG'] = two_week_tmax_avg
	covid_and_weather['2W_TMIN_AVG'] = two_week_tmin_avg
	covid_and_weather['2W_DAILY_WIND_SPEED_AVG'] = two_week_wnd_speed_avg
	covid_and_weather = covid_and_weather[['STATE_ID','STATE','COUNTY_NAME','COUNTY_FIPS','DATE','CASES','NUM_DAYS_SINCE_FIRST_CASE','NEW_CASES','PER_INCREASE_IN_CASES','2W_TMAX_AVG','2W_TMIN_AVG','2W_DAILY_WIND_SPEED_AVG']].dropna()
	covid_and_weather.to_csv("data/covid_and_weather.csv", index=False)

	covid_and_weather = pd.read_csv("data/covid_and_weather.csv")

	covid_weather_demo_data = pd.merge(covid_and_weather, demo_data[['STATE_ID', 'PERCENT_UNDER_5', 
		'PERCENT_UNDER_18', 'PERCENT_OVER_65', 'PERCENT_BELOW_POVERTY_LINE', 'POP_DENSITY']], on=['STATE_ID'], 
		how='inner')

	covid_weather_demo_mask_data = pd.merge(covid_weather_demo_data, mask_data[['STATE_ID', 
		'WEARS_MASKS_REGULARLY']], on=['STATE_ID'], how='inner')
	covid_weather_demo_mask_data.loc[covid_weather_demo_mask_data['NUM_DAYS_SINCE_FIRST_CASE'] == 0, ['NEW_CASES', 'PER_INCREASE_IN_CASES']] = 0, 0

	covid_weather_demo_mask_travel_data = pd.merge(covid_weather_demo_mask_data, travel_data[['DATE', 'COUNTY_FIPS', 'POP_AT_HOME', 'POP_NOT_AT_HOME', 
		'NUM_OF_TRIPS']], on=['DATE', 'COUNTY_FIPS'], how='inner')

	# Trim data to just required params
	final_data = covid_weather_demo_mask_travel_data[['NUM_DAYS_SINCE_FIRST_CASE','2W_TMAX_AVG','2W_TMIN_AVG',
	'2W_DAILY_WIND_SPEED_AVG','PERCENT_UNDER_5','PERCENT_UNDER_18','PERCENT_OVER_65','PERCENT_BELOW_POVERTY_LINE',
	'POP_DENSITY','WEARS_MASKS_REGULARLY','POP_AT_HOME','CASES','NEW_CASES','PER_INCREASE_IN_CASES']]
	final_data.to_csv("data/final_data.csv", index=False)
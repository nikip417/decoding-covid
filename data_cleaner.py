import weather_stations
import numpy as np
import pandas as pd
import os

weather_stations = weather_stations.weather_stations

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
	# # organize pop density data
	# format_pop_density("data/raw_data/pop_density.txt")

	# # extract state data from weather data
	# extract_state_data("data/raw_data/raw_raw_weather_data")

	# # Trim and extract relevant data
	# for file in os.listdir("data/raw_data/raw_weather_data"):
	# 	file_path = os.path.abspath(os.path.join("data/raw_data/raw_weather_data", file))
	# 	format_weather_data(file_path)

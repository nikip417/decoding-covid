# import relevant libries
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor

all_results = {}
eps = np.finfo(float).eps

def mse(predicted_y, test_y):
    e = predicted_y - test_y
    err = e ** 2 
    error = (np.sum(err) / predicted_y.shape[0])
    return error

def mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true+eps))) * 100

def normalize_data(X_train, X_test):
	for col in range(X_train.shape[1]):
		X_train[:, col] -= X_train[:, col].mean()
		X_train[:, col] /= X_train[:, col].std(ddof=1)
		
		X_test[:, col] -= X_train[:, col].mean()
		X_test[:, col] /= X_train[:, col].std(ddof=1)

	return X_train, X_test

def calculate_errors(results, X, y):
	# randomize the data, split into testing and training data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

	# normalize the data
	X_train, X_test = normalize_data(X_train, X_test)

	# add the bias term
	X_train_bias = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
	X_test_bias = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
	y_train = y_train.ravel()
	

	# train the data on different models
	# Linear Regression, , K-nearest Neighbors, Decision Trees, Random forest
	results['linear_regression'] = {}
	lr_reg = LinearRegression().fit(X_train_bias, y_train)
	y_predicted = lr_reg.predict(X_test_bias)

	reg_res = results['linear_regression']
	reg_res['mae'] = mae(y_test, y_predicted)
	reg_res['mse'] = mse(y_predicted, y_test)
	reg_res['rmse'] = mse(y_predicted, y_test) ** (1/2)
	reg_res['r2'] = r2(y_test, y_predicted)
	reg_res['mape'] = mape(y_test, y_predicted)


	# Support Vector Machines
	results['svn'] = {}
	svn_reg = svm.SVR().fit(X_train_bias, y_train)
	y_predicted = svn_reg.predict(X_test_bias)

	reg_res = results['svn']
	reg_res['mae'] = mae(y_test, y_predicted)
	reg_res['mse'] = mse(y_predicted, y_test)
	reg_res['rmse'] = mse(y_predicted, y_test) ** (1/2)
	reg_res['r2'] = r2(y_test, y_predicted)
	reg_res['mape'] = mape(y_test, y_predicted)

	
	# K-nearest Neighbors
	# TODO: figure out what k to use - I feel like the highest, I forget why we normally go with less
	# oh yea because data
	results['knn'] = {}
	for i in range(1, 11):
		results['knn'][i] = {}
		knn_reg = KNeighborsRegressor(n_neighbors=i).fit(X_train_bias, y_train)
		y_predicted = knn_reg.predict(X_test_bias)

		reg_res = results['knn'][i]
		reg_res['mae'] = mae(y_test, y_predicted)
		reg_res['mse'] = mse(y_predicted, y_test)
		reg_res['rmse'] = mse(y_predicted, y_test) ** (1/2)
		reg_res['r2'] = r2(y_test, y_predicted)
		reg_res['mape'] = mape(y_test, y_predicted)


	# Decision Tree
	# TODO: Think about adding pruning
	results['decision_tree'] = {}
	dt_reg = DecisionTreeRegressor(random_state = 0).fit(X_train_bias, y_train)
	y_predicted = dt_reg.predict(X_test_bias)

	reg_res = results['decision_tree']
	reg_res['mae'] = mae(y_test, y_predicted)
	reg_res['mse'] = mse(y_predicted, y_test)
	reg_res['rmse'] = mse(y_predicted, y_test) ** (1/2)
	reg_res['r2'] = r2(y_test, y_predicted)
	reg_res['mape'] = mape(y_test, y_predicted)


	# Random Forest Regressor
	results['random_forest_regressor'] = {}
	rfr_reg = RandomForestRegressor(random_state = 0).fit(X_train_bias, y_train)
	y_predicted = rfr_reg.predict(X_test_bias)

	reg_res = results['random_forest_regressor']
	reg_res['mae'] = mae(y_test, y_predicted)
	reg_res['mse'] = mse(y_predicted, y_test)
	reg_res['rmse'] = mse(y_predicted, y_test) ** (1/2)
	reg_res['r2'] = r2(y_test, y_predicted)
	reg_res['mape'] = mape(y_test, y_predicted)


def main():
	# import the relevant dataset
	data = pd.read_csv("data/final_data.csv")

	# With all 10 features
	# Split into x and y
	X = data[['NUM_DAYS_SINCE_FIRST_CASE', '2W_TMAX_AVG', '2W_TMIN_AVG', '2W_DAILY_WIND_SPEED_AVG', 'PERCENT_UNDER_5', 
	'PERCENT_UNDER_18', 'PERCENT_OVER_65', 'PERCENT_BELOW_POVERTY_LINE', 'POP_DENSITY', 'WEARS_MASKS_REGULARLY', 'POP_AT_HOME']].to_numpy()
	y = data[['NEW_CASES']].to_numpy()

	all_results['all_features'] = {}
	calculate_errors(all_results['all_features'], X, y)


	# With just temp features
	# Split into x and y
	X = data[['2W_TMAX_AVG', '2W_TMIN_AVG', '2W_DAILY_WIND_SPEED_AVG']].to_numpy()
	y = data[['NEW_CASES']].to_numpy()

	all_results['just_temp'] = {}
	calculate_errors(all_results['just_temp'], X, y)


	# With just temp, no wdsp features
	# Split into x and y
	X = data[['2W_TMAX_AVG', '2W_TMIN_AVG']].to_numpy()
	y = data[['NEW_CASES']].to_numpy()

	all_results['just_temp_no_wind'] = {}
	calculate_errors(all_results['just_temp_no_wind'], X, y)


	# With just demo features
	# Split into x and y
	X = data[['PERCENT_UNDER_5', 'PERCENT_UNDER_18', 'PERCENT_OVER_65', 'PERCENT_BELOW_POVERTY_LINE', 'POP_DENSITY', 
	'WEARS_MASKS_REGULARLY']].to_numpy()
	y = data[['NEW_CASES']].to_numpy()

	all_results['just_demo'] = {}
	calculate_errors(all_results['just_demo'], X, y)


	# With just demo no age features
	# Split into x and y
	X = data[['PERCENT_BELOW_POVERTY_LINE', 'POP_DENSITY']].to_numpy()
	y = data[['NEW_CASES']].to_numpy()

	all_results['just_demo_no_age'] = {}
	calculate_errors(all_results['just_demo_no_age'], X, y)


	# With just pd features
	# Split into x and y
	X = data[['POP_DENSITY']].to_numpy()
	y = data[['NEW_CASES']].to_numpy()

	all_results['just_pd_age'] = {}
	calculate_errors(all_results['just_pd_age'], X, y)


	# With just cultural features
	# Split into x and y
	X = data[['WEARS_MASKS_REGULARLY', 'POP_AT_HOME']].to_numpy()
	y = data[['NEW_CASES']].to_numpy()

	all_results['just_cult'] = {}
	calculate_errors(all_results['just_cult'], X, y)


	# With just mask features
	# Split into x and y
	X = data[['WEARS_MASKS_REGULARLY']].to_numpy()
	y = data[['NEW_CASES']].to_numpy()

	all_results['just_mask'] = {}
	calculate_errors(all_results['just_mask'], X, y)

	# With just travel features
	# Split into x and y
	X = data[['POP_AT_HOME']].to_numpy()
	y = data[['NEW_CASES']].to_numpy()

	all_results['just_travel'] = {}
	calculate_errors(all_results['just_travel'], X, y)

	print(all_results)


if __name__ == '__main__':
	main()

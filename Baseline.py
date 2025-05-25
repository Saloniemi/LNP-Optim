from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import Preprocessing
import Evaluations
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import statistics
import numpy as np

def random_forest(data, num_estimators=100):
    """Trains a Random Forest Regressor on the given data and evaluates its performance. Prints out cross valid scores and returns predictions"""
    x, y = Preprocessing.split_data_log(data)
    x_train, x_test, y_train, y_test = Preprocessing.train_test_split_data(x, y)
    rfr = RandomForestRegressor(num_estimators, random_state=42)
    rfr.fit(x_train, y_train)
    predictions = rfr.predict(x_test)
    Evaluations.evaluate_model_crossvalid(rfr, x, y)
    Evaluations.plot_predictions('Random Forest', 'Scatter', y_test, predictions)
    Evaluations.save_plot('Random Forest', 'Scatter')
    
    return predictions, rfr # saves all trained params and hyperparams, except preprocessing pipelines and data


def multi_random_forest(data, num_estimators=100):
    """Trains a multioutput Random Forest Regressor on the given data and evaluates its performance. Prints out cross valid scores and returns predictions"""
    x, y = Preprocessing.split_data_log(data)
    x_train, x_test, y_train, y_test = Preprocessing.train_test_split_data(x, y)
    rfr = RandomForestRegressor(num_estimators, random_state=42)
    multi_rfr = MultiOutputRegressor(rfr)
    multi_rfr.fit(x_train, y_train)
    predictions = multi_rfr.predict(x_test)
    Evaluations.evaluate_model_crossvalid(multi_rfr, x, y)
    Evaluations.plot_predictions('Multi Random Forest', 'Scatter', y_test, predictions)
    Evaluations.save_plot('Multi Random Forest', 'Scatter')
    
    return predictions, multi_rfr # saves all trained params and hyperparams, except preprocessing pipelines and data


def mean(data):
    """Calculates the mean of the given data. Prints outs scores and returns mean prediction"""
    mean_efficacy = statistics.mean(data['Transfection Efficacy'])
    mean_efficacy = np.full_like(np.asarray(data['Transfection Efficacy']), mean_efficacy, dtype=np.float64)
    Evaluations.evaluate_model(data['Transfection Efficacy'], mean_efficacy)
    Evaluations.plot_predictions('Mean', 'Scatter', data['Transfection Efficacy'], mean_efficacy)
    Evaluations.save_plot('Mean', 'Scatter')
    
    return mean_efficacy


def xgboost(data):
    '''Trains a multioutput xgboost model'''
    x, y = Preprocessing.split_data_log(data)
    x_train, x_test, y_train, y_test = Preprocessing.train_test_split_data(x, y)
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    xgb_model.fit(x_train, y_train)
    predictions = xgb_model.predict(x_test)
    Evaluations.evaluate_model_crossvalid(xgb_model, x, y)
    Evaluations.plot_predictions('XGBoost', 'Scatter', y_test, predictions)
    Evaluations.save_plot('XGBoost', 'Scatter')

    return predictions, xgb_model # saves all trained params and hyperparams, except preprocessing pipelines and data
    
def multi_xgboost(data):
    """Trains a multioutput XGBoost Regressor on the given data and evaluates its performance. Prints out cross valid scores and returns predictions"""
    x, y = Preprocessing.split_data_log(data)
    x_train, x_test, y_train, y_test = Preprocessing.train_test_split_data(x, y)
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    multi_xgb = MultiOutputRegressor(xgb_model)
    multi_xgb.fit(x_train, y_train)
    predictions = multi_xgb.predict(x_test)
    Evaluations.evaluate_model_crossvalid(multi_xgb, x, y)
    Evaluations.plot_predictions('Multi XGBoost', 'Scatter', y_test, predictions)
    Evaluations.save_plot('Multi XGBoost', 'Scatter')
    
    return predictions, multi_xgb # saves all trained params and hyperparams, except preprocessing pipelines and data


def linear_regression(data):
    """Trains a linear regression model on the given data and evaluates its performance. Prints out cross valid scores and returns predictions"""
    x, y = Preprocessing.split_data_log(data)
    x_train, x_test, y_train, y_test = Preprocessing.train_test_split_data(x, y)
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    predictions = lr.predict(x_test)
    Evaluations.evaluate_model_crossvalid(lr, x, y)
    Evaluations.plot_predictions('Linear Regression', 'Scatter', y_test, predictions)
    Evaluations.save_plot('Linear Regression', 'Scatter')
    
    return predictions, lr # saves all trained params and hyperparams, except preprocessing pipelines and data


def multi_linear_regression(data):
    """Trains a multioutput linear regression model on the given data and evaluates its performance. Prints out cross valid scores and returns predictions"""
    x, y = Preprocessing.split_data_log(data)
    x_train, x_test, y_train, y_test = Preprocessing.train_test_split_data(x, y)
    lr = LinearRegression()
    multi_lr = MultiOutputRegressor(lr)
    multi_lr.fit(x_train, y_train)
    predictions = multi_lr.predict(x_test)
    Evaluations.evaluate_model_crossvalid(multi_lr, x, y)
    Evaluations.plot_predictions('Multi Linear Regression', 'Scatter', y_test, predictions)
    Evaluations.save_plot('Multi Linear Regression', 'Scatter')
    
    return predictions, multi_lr # saves all trained params and hyperparams, except preprocessing pipelines and data
    
def ridge_regression(data): # same results as linear regression
    '''Trains a ridge regression model on the given data and evaluates its performance. Prints out cross valid scores and returns predictions'''
    x, y = Preprocessing.split_data_log(data)
    x_train, x_test, y_train, y_test = Preprocessing.train_test_split_data(x, y)
    rr = Ridge(alpha=1)
    rr.fit(x_train, y_train)
    predictions = rr.predict(x_test)
    Evaluations.evaluate_model_crossvalid(rr, x, y)
    Evaluations.plot_predictions('Ridge Regression', 'Scatter', y_test, predictions)
    Evaluations.save_plot('Ridge Regression', 'Scatter')

    return predictions, rr # saves all trained params and hyperparams, except preprocessing pipelines and data


def polynomial_regression(data, degree=2):
    """Trains a polynomial regression model on the given data and evaluates its performance. Prints out cross valid scores and returns predictions"""
    x, y = Preprocessing.split_data_log(data)
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(x)
    lr = LinearRegression()
    lr.fit(x_poly, y)
    predictions = lr.predict(x_poly)
    Evaluations.evaluate_model_crossvalid(lr, x_poly, y)
    Evaluations.plot_predictions('Polynomial Regression', 'Scatter', y, predictions)
    Evaluations.save_plot('Polynomial Regression', 'Scatter')

    return predictions, lr # saves all trained params and hyperparams, except preprocessing pipelines and data

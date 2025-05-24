from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import Preprocessing
import Evaluations
from sklearn.multioutput import MultiOutputRegressor
import statistics

def multi_random_forest(data, num_estimators=100):
    """Trains a multioutput Random Forest Regressor on the given data and evaluates its performance. Prints out cross valid scores and returns predictions"""
    x, y = Preprocessing.split_data(data)
    x_train, x_test, y_train, y_test = Preprocessing.train_test_split_data(x, y)
    rfr = RandomForestRegressor(num_estimators, random_state=42)
    multi_rfr = MultiOutputRegressor(rfr)
    multi_rfr.fit(x_train, y_train)
    predictions = multi_rfr.predict(x_test)
    Evaluations.evaluate_model_crossvalid(multi_rfr, x, y)
    Evaluations.plot_predictions('Random Forest', 'Scatter', y_test, predictions)
    Evaluations.save_plot('Random Forest', 'Scatter')
    
    return predictions, multi_rfr # saves all trained params and hyperparams, except preprocessing pipelines and data


def mean(data):
    """Calculates the mean of the given data. Prints outs scores and returns mean prediction"""
    mean_efficacy = statistics.mean(data['Transfection Efficacy'])
    Evaluations.evaluate_model(data['Transfection Efficacy'], mean_efficacy)
    
    return mean_efficacy


def xgboost(data):
    '''Trains a multioutput xgboost model'''
    x, y = Preprocessing.split_data(data)
    x_train, x_test, y_train, y_test = Preprocessing.train_test_split_data(x, y)
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    multi_xgb = MultiOutputRegressor(xgb_model)
    multi_xgb.fit(x_train, y_train)
    predictions = multi_xgb.predict(x_test)
    Evaluations.evaluate_model_crossvalid(multi_xgb, x, y)
    Evaluations.plot_predictions('XGBoost', 'Scatter', y_test, predictions)
    Evaluations.save_plot('XGBoost', 'Scatter')

    return predictions, multi_xgb # saves all trained params and hyperparams, except preprocessing pipelines and data
    

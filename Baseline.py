from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import Preprocessing
import Evaluations
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import statistics
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import norm
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

## AI GENERATED (NEED TO BE TESTED + MODIFIED)
def fit_naive_gaussian_regression(X_train, y_train, n_bins=10):
    """
    Fits a naive Gaussian Bayes regression model.

    Parameters:
        X_train: ndarray, shape (n_samples, n_features)
        y_train: ndarray, shape (n_samples,)
        n_bins: int, number of bins to discretize y

    Returns:
        model: dict with keys:
            - 'feature_stats': {class: (means, stds)}
            - 'class_priors': {class: prior probability}
            - 'bin_means': mean value of each bin (to reconstruct y)
            - 'est': the fitted KBinsDiscretizer object
    """
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    y_binned = est.fit_transform(y_train.reshape(-1, 1)).astype(int).ravel()

    feature_stats = {}
    class_priors = {}

    for c in np.unique(y_binned):
        X_c = X_train[y_binned == c]
        means = X_c.mean(axis=0)
        stds = X_c.std(axis=0) + 1e-6  # Avoid divide by zero
        feature_stats[c] = (means, stds)
        class_priors[c] = np.mean(y_binned == c)

    # Mean value of y in each bin (expected y)
    bin_means = est.bin_edges_[0].mean(axis=1)

    return {
        'feature_stats': feature_stats,
        'class_priors': class_priors,
        'bin_means': bin_means,
        'est': est
    }


def predict_naive_gaussian_regression(X, model):
    """
    Predicts using the fitted naive Gaussian regression model.

    Parameters:
        X: ndarray, shape (n_samples, n_features)
        model: dict returned by fit_naive_gaussian_regression

    Returns:
        y_pred: ndarray, shape (n_samples,)
    """
    feature_stats = model['feature_stats']
    class_priors = model['class_priors']
    bin_means = model['bin_means']
    n_bins = len(bin_means)

    y_pred = []
    for x in X:
        probs = []
        for c in range(n_bins):
            mean, std = feature_stats[c]
            likelihood = np.prod(norm.pdf(x, mean, std))
            prior = class_priors[c]
            probs.append(likelihood * prior)
        probs = np.array(probs)
        probs /= probs.sum()  # Normalize to get P(y|x)
        pred = np.sum(probs * bin_means)
        y_pred.append(pred)

    return np.array(y_pred)
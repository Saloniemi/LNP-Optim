from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score, cross_val_predict
import statistics
import numpy as np
import datetime 
import os

def evaluate_model_crossvalid(model, x, y):
    """Evaluates the model using cross-validation. Prints out scores and returns them"""
    cv = 5
    # MSE
    mse_scores = cross_val_score(model, x, y, cv=cv, scoring='neg_mean_squared_error')
    mse_cv = -statistics.mean(mse_scores)
    mse_cv_std = statistics.stdev(mse_scores)
    # R^2
    r2_scores = cross_val_score(model, x, y, cv=cv, scoring='r2')
    r2_cv = statistics.mean(r2_scores)
    r2_cv_std = statistics.stdev(r2_scores)
    # Pearson correlation
    pred = cross_val_predict(model, x, y, cv=cv)
    corr,_ = pearsonr(pred.ravel(), y.values.ravel())
    print(f'CV Mean Squared Error: {mse_cv}, std: {mse_cv_std}')
    print(f'CV R^2 scores: {r2_cv}, std: {r2_cv_std}')
    print(f'CV Pearson correlation: {corr}')

    return mse_cv, r2_cv, corr


def evaluate_model(y_test, predictions):
    """Evaluates the model using Mean Squared Error and R^2 score. Prints out scores and returns them"""
    # if shape doesn't match (i.e. a single mean value as prediction)
    if len(y_test.shape) != len(predictions.shape):
        predictions = np.full_like(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    corr,_ = pearsonr(y_test, predictions)
    print(f'Mean Squared Error: {mse}')
    print(f'R^2: {r2}')
    print(f'Pearson correlation: {corr}')
    
    return mse, r2


def plot_predictions(name, mode, y_test, predictions):
    """Plots the predictions against the true values by different modes."""
    if mode == 'Scatter':
        plt.scatter(predictions, y_test, label='True Values', color='blue')
        plt.title(f'{name} {mode} Predictions vs True Values')


def save_plot(name, mode):
    """Saves the plot to a file"""
    save_dir = 'plots'
    os.makedirs(save_dir, exist_ok=True)
    now = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    filename = os.path.join(save_dir, f"{name}_{mode}_{now} Plot_.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score, cross_val_predict
import statistics
import numpy as np
import datetime 
import os

def data_visualization(data):
    '''Visualizes the data distribution using a histogram, scatter plot, and box plot.'''
    indep = 'Transfection Efficacy'
    dep1 = 'Ion/Helper Percent'
    dep2 = 'N/P Ratio'
    dep3 = 'Ion/Helper Ratio'
    dep4 = 'Chol/PEG Ratio'

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    # Histogram
    axes[0,0].hist(data[indep], bins=30, color='blue', alpha=0.7)
    axes[0,0].set_title(f'{indep} Distribution')
    axes[0,0].set_xlabel(indep)
    axes[0,0].set_ylabel('Frequency')
    # Scatter plot 1
    axes[0,1].scatter(data[dep1], data[indep], color='blue', alpha=0.7)
    axes[0,1].set_title(f'{indep} vs {dep1}')
    axes[0,1].set_xlabel(dep1)
    axes[0,1].set_ylabel(indep)
    # Scatter plot 2
    axes[1,0].scatter(data[dep2], data[indep], color='blue', alpha=0.7)
    axes[1,0].set_title(f'{indep} vs {dep2}')
    axes[1,0].set_xlabel(dep2)
    axes[1,0].set_ylabel(indep)
    # Scatter plot 3
    axes[1,1].scatter(data[dep4], data[indep], color='blue', alpha=0.7)
    axes[1,1].set_title(f'{indep} vs {dep4}')
    axes[1,1].set_xlabel(dep4)
    axes[1,1].set_ylabel(indep)
    save_plot('Data Visualization', 'Histogram and Scatter')
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    axes[0,0].hist(data[dep1], bins=30, color='blue', alpha=0.7)
    axes[0,0].set_title(f'{dep1} Distribution')
    axes[0,1].hist(data[dep2], bins=30, color='blue', alpha=0.7)
    axes[0,1].set_title(f'{dep2} Distribution')
    axes[1,0].hist(data[dep3], bins=30, color='blue', alpha=0.7)
    axes[1,0].set_title(f'{dep3} Distribution')
    axes[1,1].hist(data[dep4], bins=30, color='blue', alpha=0.7)
    axes[1,1].set_title(f'{dep4} Distribution')
    save_plot('Data Visualization', 'Histogram')
    plt.show()



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
    corr,_ = pearsonr(pred.ravel(), y)
    print(f'CV Mean Squared Error: {mse_cv}, std: {mse_cv_std}')
    print(f'CV R^2 scores: {r2_cv}, std: {r2_cv_std}')
    print(f'CV Pearson correlation: {corr}')

    return mse_cv, r2_cv, corr


def evaluate_model(y_test, predictions):
    """Evaluates the model using Mean Squared Error and R^2 score. Prints out scores and returns them"""
    # if prediction is a single mean value
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
        fig, ax = plt.subplots(2)
        ax[0].scatter(y_test, predictions, label='True Values', color='blue')
        ax[0].axline((np.min(y_test),np.min(predictions)), slope = 1, color='red', linestyle='--')
        ax[0].set_title(f'{name} {mode} Predictions vs True Values')
        ax[0].set_xlabel('True Values')
        ax[0].set_ylabel('Predictions')
        ax[1].hist(predictions, bins=30, color='blue', alpha=0.7)
        ax[1].set_title(f'{name} {mode} Predictions Distribution')
        ax[1].set_xlabel('Predictions')
        ax[1].set_ylabel('Frequency')


def save_plot(name, mode):
    """Saves the plot to a file"""
    save_dir = 'plots'
    os.makedirs(save_dir, exist_ok=True)
    now = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    filename = os.path.join(save_dir, f"{name}_{mode}_{now} Plot_.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
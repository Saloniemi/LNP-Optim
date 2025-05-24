from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr
import statistics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

rfr = RandomForestRegressor(n_estimators=100, random_state=42)

multi_rfr = MultiOutputRegressor(rfr)

multi_rfr.fit(x_train, y_train)

predictions = multi_rfr.predict(x_test)

mse_cv = -statistics.mean(cross_val_score(multi_rfr, x, y, cv=5, scoring='neg_mean_squared_error'))
r2_cv = statistics.mean(cross_val_score(multi_rfr, x, y, cv=5, scoring='r2'))
print(f'Cross-validated Mean Squared Error: {mse_cv}')
print(f'Cross-validated R^2 scores: {r2_cv}')

corr,_ = pearsonr(y_test.values.flatten(), predictions.flatten())
print(corr)

plt.scatter(predictions,y_test, label='True Values', color='blue')
plt.title('Random Forest Regressor Predictions vs True Values')
save_dir = "plots"
os.makedirs(save_dir, exist_ok=True)
filename = os.path.join(save_dir, "Random_Forest_Plot_1.png")
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.show()
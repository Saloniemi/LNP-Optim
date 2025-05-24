import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Loads and preprocesses the data from the given file path. Returns data"""
    data = pd.read_csv(file_path)

    return data

    
def split_data(data):
    """Splits the data into features and target variable. Returns x and y"""
    x = data[['Ion/Helper Percent', 'N/P Ratio', 'Ion/Helper Ratio', 'Chol/PEG Ratio']]
    y = np.log1p(data[['Transfection Efficacy']]) 

    return x, y


def train_test_split_data(x, y):
    """Splits the data into training and testing sets. Returns x_train, x_test, y_train, y_test"""
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    return x_train, x_test, y_train, y_test


'''
*******************************
Author:
u3257896,u3280573,u3241996,u3273874 4483 Undergraduate Online 2 Assessment 3  20/10/2024
Boat Price Prediction
*******************************
'''

import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import joblib

class BoatModel:
    def __init__(self):
        """ Initialize the BoatModel with default parameters and models. """
        self.data = None
        self.target_variable = 'price'
        #Lists the models
        self.models = [
            ('Linear Regression', LinearRegression()),
            ('Decision Tree', DecisionTreeRegressor()),
            ('Random Forest', RandomForestRegressor()),
            ('AdaBoost', AdaBoostRegressor()),
            ('XGBoost', XGBRegressor()),
            ('KNN', KNeighborsRegressor()),
            ('SVM', SVR())
        ]

    def load_data(self, file_path):
        """ Load dataset from a specified file path. """
        print("Loading dataset...")
        self.data = pd.read_csv(file_path)
        print("Dataset loaded")

    def clean_data(self):
        """ Clean the data by removing duplicates, handling missing values, and removing outliers. """
        print("Cleaning data...")
        #Removes the duplicates
        self.data = self.data.drop_duplicates()
        #Removes the rows with missing values
        self.data = self.data.dropna()
        #Removes the outliers
        z_scores = stats.zscore(self.data.select_dtypes(include=['int64', 'float64']))
        abs_z_scores = abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        self.data = self.data[filtered_entries]
        print("Data cleaned")

    def eda(self, data_type):
        """ Perform Exploratory Data Analysis (EDA). Visualize categorical or continuous data based on the provided data type. """
        print("Performing EDA...")
        if data_type == "Categorical":
            #Creates the categorical variables
            categorical_columns = self.data.select_dtypes(include=['object']).columns
            categorical_columns = categorical_columns.drop(['id', 'boatClass', 'make','model','city','state','zip','created_date'], errors='ignore')
            for column in categorical_columns:
                plt.figure(figsize=(10, 5))
                sns.countplot(data=self.data, x=column)
                plt.title(f'Distribution of {column}')
                plt.show()
        elif data_type == "Continuous":
            #creates the continuous variables
            quantitative_columns = self.data.select_dtypes(include=['int64', 'float64']).columns
            quantitative_columns = quantitative_columns.drop(['id'], errors='ignore')
            for column in quantitative_columns:
                plt.figure(figsize=(10, 5))
                sns.histplot(self.data[column], kde=True)
                plt.title(f'Distribution of {column}')
                plt.xlabel(column)
                plt.ylabel('Frequency')
                plt.show()

    def handle_missing_values(self):
        """ Handle missing values in the dataset by imputing them with median (for continuous variables) or mode (for categorical variables). """
        print("Handling missing values...")
        quantitative_columns = self.data.select_dtypes(include=['int64', 'float64']).columns
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        #Assigns the missing values for the continuous variables with median
        for column in quantitative_columns:
            median = self.data[column].median()
            self.data[column] = self.data[column].fillna(median)
        #Assigns the missing values for categorical variables with mode
        for column in categorical_columns:
            mode = self.data[column].mode()[0]
            self.data[column] = self.data[column].fillna(mode)
        print("Missing values handled")

    def feature_selection(self):
        """ Select important features for the model by analyzing their relationship with the target variable. """
        print("Selecting features...")
        #Creates a scatter plot continuous vs continuous
        quantitative_columns = self.data.select_dtypes(include=['int64', 'float64']).columns
        for column in quantitative_columns:
            if column != self.target_variable:
                plt.figure(figsize=(10, 5))
                sns.scatterplot(x=self.data[column], y=self.data[self.target_variable])
                plt.title(f'Scatter plot between {column} and {self.target_variable}')
                plt.show()
        correlation_matrix = self.data.corr()
        print("Correlation Matrix:")
        print(correlation_matrix[self.target_variable].sort_values(ascending=False))
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            model = ols(f'{self.target_variable} ~ C({column})', data=self.data).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            print(f"ANOVA results for {column}:\n", anova_table)

    def convert_to_numeric(self):
        """ Convert categorical variables to numeric using one-hot encoding. """
        print("Converting to numeric...")
        self.data = pd.get_dummies(self.data, drop_first=True)
        print("Conversion to numeric complete")

    def train_test_split(self):
        """ Split the data into training and testing sets, and standardize the features. """
        print("Splitting data...")
        if self.data is None:
            print("Data not loaded!")
            return None, None, None, None
        X = self.data.drop(columns=[self.target_variable])
        y = self.data[self.target_variable]
        #Splits the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test

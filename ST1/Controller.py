'''
*******************************
Author:
u3257896,u3280573,u3241996,u3273874 4483 Undergraduate Online 2 Assessment 3  20/10/2024
Boat Price Prediction
*******************************
'''

from Model import BoatModel

class BoatController:
    def __init__(self):
        self.model = BoatModel()

    def load_data(self, file_path):
        self.model.load_data(file_path)

    def clean_data(self):
        self.model.clean_data()

    def eda(self, data_type, *args):
        self.model.eda(data_type)

    def handle_missing_values(self):
        self.model.handle_missing_values()

    def feature_selection(self):
        self.model.feature_selection()

    def convert_to_numeric(self):
        self.model.convert_to_numeric()

    def train_test_split(self):
        if self.model.data is None:
            print("Data not loaded!")
            return None, None, None, None
        return self.model.train_test_split()

    def evaluate_models(self, X_train, X_test, y_train, y_test):
        return self.model.evaluate_models(X_train, X_test, y_train, y_test)

    def save_model(self, model_name):
        self.model.save_model(model_name)

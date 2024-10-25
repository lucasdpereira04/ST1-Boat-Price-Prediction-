'''
*******************************
Author:
u3257896,u3280573,u3241996,u3273874 4483 Undergraduate Online 2 Assessment 3  20/10/2024
Boat Price Prediction
*******************************
'''

from Controller import BoatController
from View import create_gui

if __name__ == "__main__":
    print("Starting application...")
    controller = BoatController()
    create_gui(controller)
    print("Application running...")


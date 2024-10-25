'''
*******************************
Author:
u3257896,u3280573,u3241996,u3273874 4483 Undergraduate Online 2 Assessment 3  20/10/2024
Boat Price Prediction
*******************************
'''

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from Controller import BoatController  #Import statement is added


class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        label = ttk.Label(self.tooltip, text=self.text, background="#ffffe0", relief="solid", borderwidth=1)
        label.pack()

    def hide_tooltip(self, event):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


def create_gui(controller):
    def on_closing():
        root.destroy()

    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", on_closing)  # Handle the window close event
    root.title("Boat Information System")
    root.geometry("400x600")
    root.configure(bg="#f0f0f0")

    title_label = tk.Label(root, text="Boat Information System", font=("Arial", 18, "bold"), bg="#f0f0f0")
    title_label.pack(pady=10)

    button_style = {"font": ("Arial", 12), "bg": "#808080", "fg": "white", "relief": "raised", "bd": 3}

    def load_data_callback():
        file_path = filedialog.askopenfilename()
        controller.load_data(file_path)

    def clean_data_callback():
        controller.clean_data()

    def eda_callback():
        data_type = data_type_var.get()
        controller.eda(data_type)

    def handle_missing_callback():
        controller.handle_missing_values()

    def feature_selection_callback():
        controller.feature_selection()

    def convert_numeric_callback():
        controller.convert_to_numeric()

    def train_test_split_callback():
        X_train, X_test, y_train, y_test = controller.train_test_split()
        if None not in (X_train, X_test, y_train, y_test):
            best_model_name = controller.evaluate_models(X_train, X_test, y_train, y_test)
            controller.save_model(best_model_name)
        else:
            print("Train/Test split failed. Please check your data.")

    data_type_var = tk.StringVar(root)
    data_type_var.set("Choose Data Type")
    data_type_menu = tk.OptionMenu(root, data_type_var, "Categorical", "Continuous")
    data_type_menu.config(font=("Arial", 12), bg="#808080", fg="white", relief="raised", bd=3)
    data_type_menu.pack(pady=10)
    ToolTip(data_type_menu, "Select the type of data to visualize")

    load_data_btn = tk.Button(root, text="Load Data", command=load_data_callback, **button_style)
    load_data_btn.pack(pady=10)
    ToolTip(load_data_btn, "Load your dataset file")

    clean_data_btn = tk.Button(root, text="Clean Data", command=clean_data_callback, **button_style)
    clean_data_btn.pack(pady=10)
    ToolTip(clean_data_btn, "Remove duplicates and handle missing values")

    eda_btn = tk.Button(root, text="Exploratory Data Analysis", command=eda_callback, **button_style)
    eda_btn.pack(pady=10)
    ToolTip(eda_btn, "Visualize the distribution of the chosen data type")

    handle_missing_btn = tk.Button(root, text="Handle Missing Values", command=handle_missing_callback, **button_style)
    handle_missing_btn.pack(pady=10)
    ToolTip(handle_missing_btn, "Impute missing values in the dataset")

    feature_selection_btn = tk.Button(root, text="Feature Selection", command=feature_selection_callback,
                                      **button_style)
    feature_selection_btn.pack(pady=10)
    ToolTip(feature_selection_btn, "Perform feature selection based on correlation and ANOVA")

    convert_numeric_btn = tk.Button(root, text="Convert to Numeric", command=convert_numeric_callback, **button_style)
    convert_numeric_btn.pack(pady=10)
    ToolTip(convert_numeric_btn, "Convert categorical variables to numeric using one-hot encoding")

    train_test_split_btn = tk.Button(root, text="Train/Test Split and Model Evaluation",
                                     command=train_test_split_callback, **button_style)
    train_test_split_btn.pack(pady=10)
    ToolTip(train_test_split_btn, "Split the data, train models, and evaluate their performance")

    root.mainloop()


controller = BoatController()
create_gui(controller)

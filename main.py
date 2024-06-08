import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import tkinter as tk
from tkinter import ttk
import warnings

warnings.filterwarnings('ignore')
dataset = pd.read_csv("heart.csv")

info = ["age", "1: male, 0: female",
        "chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic",
        "resting blood pressure", " serum cholestoral in mg/dl", "fasting blood sugar > 120 mg/dl",
        "resting electrocardiographic results (values 0,1,2)", " maximum heart rate achieved",
        "exercise induced angina", "oldpeak = ST depression induced by exercise relative to rest",
        "the slope of the peak exercise ST segment", "number of major vessels (0-3) colored by flourosopy",
        "thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]


y = dataset["target"]

sns.countplot(y)

target_temp = dataset.target.value_counts()

sns.countplot(x="sex", hue="target", data=dataset)
# plt.show()

predictors = dataset.drop("target", axis=1)
target = dataset["target"]

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=42)

# Train the Random Forest Classifier with random_state=42
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred_rf = rf.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(Y_pred_rf, Y_test)

# Feature names
feature_labels = ["Age", "Sex", "Chest Pain Type", "Resting Blood Pressure", "Cholesterol",
                  "Fasting Blood Sugar", "Resting ECG", "Max Heart Rate", "Exercise Induced Angina",
                  "Oldpeak", "Slope", "Number of Major Vessels", "Thal"]

inuput_method = ["e", "d", "d", "e", "e", "e", "d", "e", "e", "e", "e", "d", "d"]

options = [
    [],
    ["Female", "Male"],
    ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"],
    [],
    [],
    [],
    ["0", "1", "2"],
    [],
    [],
    [],
    [],
    ["0", "1", "2", "3"],
    ["normal", "fixed defect", "reversable defect"]
]

def predict_result():
    # Get input values
    features = []
    for i, entry in enumerate(feature_entries):
        if inuput_method[i] == "d":
            features.append(float(options[i].index(entry.get())))
        else:
            features.append(float(entry.get()))

    # Make prediction
    features = np.array([features])
    result = rf.predict(features)

    # Update the result label based on the prediction
    if result[0] == 0:
        result_label.config(text="Heart Disease Not Present")
    else:
        result_label.config(text="Heart Disease Present")


# GUI setup
root = tk.Tk()
root.geometry("345x610")
root.resizable(False, False)
root.title("Heart Disease Predictor")
root.config(bg='lightblue')

# Top Frame
top_frame = ttk.Frame(root, padding="20")
top_frame.grid(row=0, column=0, padx=10, pady=10)

# Feature Entry Widgets
feature_entries = []

for i, label_text in enumerate(feature_labels):
    ttk.Label(top_frame, text=f"{label_text}:").grid(row=i, column=0, padx=5, pady=5)
    if inuput_method[i] == "e":
        entry = ttk.Entry(top_frame)
        entry.grid(row=i, column=1, padx=5, pady=5)
    else:
        entry = tk.StringVar()
        entry.set(options[i][0])
        menu = ttk.OptionMenu(top_frame, entry, "Select Option", *options[i])
        menu.grid(row=i, column=1, padx=5, pady=5)

    feature_entries.append(entry)

# Prediction Button
predict_button = ttk.Button(top_frame, text="Predict", command=predict_result)
predict_button.grid(row=len(feature_labels), column=0, columnspan=2, pady=10)

# Bottom Frame
bottom_frame = ttk.Frame(root, padding="20")
bottom_frame.grid(row=1, column=0, padx=10, pady=10)

# Result Label
result_label = ttk.Label(bottom_frame, text="Predicted Result: -")
result_label.grid(row=0, column=0, padx=5, pady=5)



root.mainloop()

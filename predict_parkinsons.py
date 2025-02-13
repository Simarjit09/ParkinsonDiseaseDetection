import joblib
import numpy as np

# Load the saved model
model = joblib.load("parkinsons_model.pkl")

# REAL PATIENT DATA
new_data = np.array([[0.184, 0.112, 0.054, 0.195, 0.249, 0.157, 0.134, 0.112, 0.169, 0.224]])

# Make prediction
prediction = model.predict(new_data)

# Output the result
if prediction[0] == 1:
    print("The model predicts the person HAS Parkinson's Disease.")
else:
    print("The model predicts the person is HEALTHY.")

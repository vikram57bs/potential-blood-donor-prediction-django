from django.shortcuts import render
import numpy as np
from .modelfile import get_trained_model
import pandas as pd  # Adjust the import according to your project structure

# Load the trained Random Forest model


def predict_donation(recency,frequency,monetary,months):
    # Prepare the input data in the same format the model was trained on
    rf_model,scaler= get_trained_model()
    input_data = np.array([[recency, frequency, monetary, months]])
    asd=input_data.reshape(1,-1)
    input_scaled = scaler.transform(asd)
    columns = ['Recency (months)', 'Frequency (times)', 'Monetary (c.c. blood)', 'Time (months)']
    input_dff = pd.DataFrame(input_scaled, columns=columns)
    prediction = rf_model.predict(input_dff)
    
    # Return "Eligible Donor" if the prediction is 1, otherwise "Not Eligible Donor"
    print(prediction[0])
    return "Not Eligible Donor" if prediction[0] == 0 else "Eligible Donor"

def predict(request):
    prediction = None

    if request.method == 'POST':
        recency = int(request.POST['recency'])
        frequency = int(request.POST['frequency'])
        monetary = int(request.POST['monetary'])
        months = int(request.POST['months'])
     # This input will not be used for prediction
     

        # Make prediction
        prediction = predict_donation(recency,frequency,monetary,months)
    
    return render(request, 'predictor/form.html', {'prediction': prediction})


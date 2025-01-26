import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
def get_trained_model():
    # Load your data
   
    data = pd.read_csv(r'C:\Users\Admin\OneDrive\Desktop\blood_donor_prediction\predictor\transfusion.csv')
    print('im running')


    # Preprocessing
    X = data[['Recency (months)', 'Frequency (times)', 'Monetary (c.c. blood)', 'Time (months)']]
    y = data['whether he/she donated blood in March 2007']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Train the Random Forest model
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train, y_train)
    #aw=np.array([2,6,1500,15])
    #aq=aw.reshape(1,-1)
#print(aq)
    #columns = ['Recency (months)', 'Frequency (times)', 'Monetary (c.c. blood)', 'Time (months)']  # Replace with the actual column names used during training
    #input_df = pd.DataFrame(aq, columns=columns)
#print(input_df)
    #asz=rf_classifier.predict(input_df)
    #print(asz[0])
    return rf_classifier,scaler
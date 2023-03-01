#Importing the required packages. 
#Ensure to install xgboost

from xgboost import XGBClassifier
from google.cloud import storage
import pandas as pd
import joblib
#picking the data transformed file from the cloud stoarge for model building.

storage_client = storage.Client()
public_bucket = storage_client.bucket('pipeline-artifacts-demo')
blob = public_bucket.blob('Transformed_data/X_transformed.csv')
blob.download_to_filename('X_transformed.csv')
X_transformed=pd.read_csv("X_transformed.csv")
blob = public_bucket.blob('train_data/y_train.csv')
blob.download_to_filename('y_train.csv')
train_Y=pd.read_csv('y_train.csv')
#Model training, no hyper parameter activity is included. 
model = XGBClassifier()
model.fit(X_transformed,train_Y)
joblib.dump(model,'trained_model.joblib')
#Pushing the built model to the artifacts. 
public_bucket.blob('Trained_model/trained_model.joblib').upload_from_filename('trained_model.joblib')
print("Model is built and stored in the artifacts")
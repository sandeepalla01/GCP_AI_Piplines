#Importing the required packages
from google.cloud import storage
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

#Since we are using cloud stoarge as the artifact storage, we need to pick the file agian from cloud storage.

storage_client = storage.Client()
public_bucket = storage_client.bucket('pipeline-artifacts-demo')
blob = public_bucket.blob('Raw_data/Bank_Churners.csv')
blob.download_to_filename('Bank_Churners.csv')
data=pd.read_csv('Bank_Churners.csv')

#making X and y to be dataframe for dependent and independent variables. 

X = data.drop(columns=['Attrition_Flag'])
y = data['Attrition_Flag']

#since we are having imbalanced dataset, stratify is set to true
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100,stratify=y)

#X_train, X_test, y_train, y_test will be the input for the next step in the pipeline, we need to push it back to the cloud storage.

storage_client = storage.Client()
public_bucket = storage_client.bucket('pipeline-artifacts-demo')
X_train.to_csv("X_train.csv",index=False)
public_bucket.blob("train_data/X_train.csv").upload_from_filename("X_train.csv", content_type='text/csv')
y_train.to_csv("y_train.csv",index=False)
public_bucket.blob("train_data/y_train.csv").upload_from_filename("y_train.csv", content_type='text/csv')
X_test.to_csv("X_test.csv",index=False)
public_bucket.blob('test_data/X_test.csv').upload_from_filename('X_test.csv', content_type='text/csv')
y_test.to_csv("y_test.csv",index=False)
public_bucket.blob('test_data/y_test.csv').upload_from_filename('y_test.csv', content_type='text/csv')
print("Data splitting successfully completed")

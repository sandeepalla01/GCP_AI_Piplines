#Importing the required packages. 
import joblib
from google.cloud import storage
import pandas as pd

#Pulling the data from the artifacts, the test data and the pipeline model for data transformation. 
storage_client = storage.Client()
public_bucket = storage_client.bucket('pipeline-artifacts-demo')
blob = public_bucket.blob('test_data/X_test.csv')
blob.download_to_filename('X_test.csv')
X_test=pd.read_csv("X_test.csv")
blob = public_bucket.blob('test_data/y_test.csv')
blob.download_to_filename('y_test.csv')
y_test=pd.read_csv("y_test.csv")
blob = public_bucket.blob('Transformed_data/transformed_steps.joblib')
blob.download_to_filename('transformed_steps.joblib')
transformed_steps=joblib.load('transformed_steps.joblib')
X_test_tmp=transformed_steps.transform(X_test)

#X_test_tmp will in the numpy format and all the column header information will be lost in the data transformation process. 
#Below steps to get back the column names of the test data
cols_categorical = ['Gender','Dependent_count', 'Education_Level', 'Marital_Status','Income_Category','Card_Category']


col_categorical=transformed_steps.named_steps['transform_column'].transformers_[2][1].named_steps['onehotencoding'].get_feature_names(cols_categorical)
col_categorical.tolist()
cols_numeric = ['Customer_Age', 'Months_on_book','Total_Relationship_Count','Months_Inactive_12_mon','Credit_Limit','Total_Revolving_Bal','Avg_Open_To_Buy','Total_Amt_Chng_Q4_Q1','Total_Trans_Amt','Total_Ct_Chng_Q4_Q1','Avg_Utilization_Ratio']
new_col_names=cols_numeric+col_categorical.tolist()
X_test_transformed=pd.DataFrame(X_test_tmp, columns=new_col_names)

#Pushing the transformed test data back to the artifacts 
X_test_transformed.to_csv("X_test_transformed.csv",index=False)
storage_client = storage.Client()
public_bucket = storage_client.bucket('pipeline-artifacts-demo')
public_bucket.blob('Transformed_data/X_test_transformed.csv').upload_from_filename('X_test_transformed.csv', content_type='text/csv')

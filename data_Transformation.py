#Importing required packages.
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import sklearn
from google.cloud import storage
import pandas as pd
import joblib

#Fetching data from the artifacts (stored in the cloud storage)
storage_client = storage.Client()
public_bucket = storage_client.bucket('pipeline-artifacts-demo')
blob = public_bucket.blob('train_data/X_train.csv')
blob.download_to_filename('X_train.csv')
X_train=pd.read_csv("X_train.csv")
blob = public_bucket.blob('train_data/y_train.csv')
blob.download_to_filename('y_train.csv')
train_Y=pd.read_csv("y_train.csv")

#creating lists to store the column names of the dataframe.
cols_drop=  ['CLIENTNUM']
cols_numeric = ['Customer_Age', 'Months_on_book','Total_Relationship_Count','Months_Inactive_12_mon','Credit_Limit','Total_Revolving_Bal','Avg_Open_To_Buy','Total_Amt_Chng_Q4_Q1','Total_Trans_Amt','Total_Ct_Chng_Q4_Q1','Avg_Utilization_Ratio']
cols_categorical = ['Gender','Dependent_count', 'Education_Level', 'Marital_Status','Income_Category','Card_Category']

#Creating a sklearn pipelines for the data transformation
transformer_numeric = Pipeline(steps=[
                                     ('num_mean_imputer', SimpleImputer(strategy='mean')),
                                     ])

#One hot encoding of the categorical columns, we have kept Dependent_count column to be categorical 
transformer_categorical = Pipeline(steps=[
                                         ('onehotencoding', OneHotEncoder(handle_unknown='ignore'))
                                         ])
transformer_column = ColumnTransformer(transformers=[('drop_columns', 'drop', cols_drop),
                                                   ('numeric_processing',transformer_numeric, cols_numeric),
                                                    ('categorical_processing', transformer_categorical, cols_categorical)
                                                  ], remainder='drop')
pipeline = Pipeline([('transform_column', transformer_column),])
pipeline.fit(X_train)
X_train_transformed=pipeline.transform(X_train)
col_categorical=pipeline.named_steps['transform_column'].transformers_[2][1].named_steps['onehotencoding'].get_feature_names(cols_categorical)
col_categorical.tolist()
new_col_names=cols_numeric+col_categorical.tolist()
X_transformed=pd.DataFrame(X_train_transformed, columns=new_col_names)
#Pushing the data transformed file and also the pipeline model which is created for data transformation. 
#The same pipeline model will be used to transform the test data 
X_transformed.to_csv("X_transformed.csv",index=False)
public_bucket = storage_client.bucket('pipeline-artifacts-demo')
public_bucket.blob('Transformed_data/X_transformed.csv').upload_from_filename('X_transformed.csv', content_type='text/csv')
joblib.dump(pipeline, 'transformed_steps.joblib')
public_bucket.blob('Transformed_data/transformed_steps.joblib').upload_from_filename('transformed_steps.joblib')
print("Transformed data and the pipeline model created and stored in the artifacts")
#Importing required packages
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.cloud import storage

#To supress the warnings
import warnings
warnings.filterwarnings('ignore')

#To fetch the data from the bigquery. 
#Data will be stored back to the cloud storage as the csv file.
bqclient = bigquery.Client()
query_string = 'SELECT * FROM  `ai-platform-01.bank_churn.bank_churners_data`'
Bank_Churners = (
    bqclient.query(query_string)
    .result()
    .to_dataframe(
        create_bqstorage_client=True,
    )
)
Bank_Churners.to_csv("Bank_Churners.csv",index=False)
storage_client = storage.Client()
public_bucket = storage_client.bucket('pipeline-artifacts-demo')
public_bucket.blob('Raw_data/Bank_Churners.csv').upload_from_filename('Bank_Churners.csv', content_type='text/csv')
print("data loaded successfully")
#Importing the required packages. 
import joblib
from google.cloud import storage
import pandas as pd
import os
from sklearn.metrics import classification_report, confusion_matrix
import json
import numpy as np
#We are not using the tensorflow package for model building but to write to a json file. 
import tensorflow as tf
from tensorflow.python.lib.io import file_io 

#Pulling the transformed test data and the trained model from the artifacts 
storage_client = storage.Client()
public_bucket = storage_client.bucket('pipeline-artifacts-demo')
blob = public_bucket.blob('Transformed_data/X_test_transformed.csv')
blob.download_to_filename('X_test_transformed.csv')
X_test_transformed=pd.read_csv("X_test_transformed.csv")
blob = public_bucket.blob('test_data/y_test.csv')
blob.download_to_filename('y_test.csv')
y_test=pd.read_csv("y_test.csv")
blob = public_bucket.blob('Trained_model/trained_model.joblib')
blob.download_to_filename('trained_model.joblib')
trained_model=joblib.load('trained_model.joblib')

#Testing the trained model on the transformed test data
trained_model.score(X_test_transformed,y_test)
y_predicted = trained_model.predict(X_test_transformed)
confusion_matrix(y_test, y_predicted)
classification_report(y_test, y_predicted, labels=['Existing Customer','Attrited Customer'])
tn, fp, fn, tp = confusion_matrix(y_test, y_predicted).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp + fn)
acc=(tp+tn)/(tp+tn+fp+fn)

#writing the evaluation metrics into Json file to see it in the pipeline results.
with tf.io.gfile.GFile("gs://pipeline-artifacts-demo/Evaluation"+ "/metrics.json", 'w') as outfile:
    json.dump({ "accuracy": acc, "specificity": specificity, "sensitivity":sensitivity}, outfile)
vocab = list(np.unique(y_test))

#Storing the confusion matrix in the csv file to store in artifacts. 
tmp_list=[]
cm = confusion_matrix(y_test, y_predicted, labels=vocab)
for target_index, target_row in enumerate(cm):
    for predicted_index, count in enumerate(target_row):
        tmp_list.append((vocab[target_index], vocab[predicted_index], count))
df_cm = pd.DataFrame(tmp_list, columns=['target', 'predicted', 'count'])

cm_file = os.path.join("gs://pipeline-artifacts-demo/Evaluation/", 'confusion_matrix.csv')
with file_io.FileIO(cm_file, 'w') as f:
    df_cm.to_csv(f, columns=['target', 'predicted', 'count'], header=True, index=False)

metrics = {
    'metrics': [{
      'name': 'accuracy-score',
      'numberValue':  acc,
      'format': "PERCENTAGE",
    }, 
    {
      'name': 'specificity',
      'numberValue':  specificity,
      'format': "PERCENTAGE",
    }, 
    {
      'name': 'sensitivity',
      'numberValue':  sensitivity,
      'format': "PERCENTAGE",
    },]
  }
with file_io.FileIO('/mlpipeline-metrics.json', 'w') as f:
    json.dump(metrics, f)
print("Model tested on the test data successfully")
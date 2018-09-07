# [START setup]
import datetime
import os
import pandas as pd
import subprocess

from google.cloud import storage

from sklearn.ensemble import KMeans
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer


# TODO: REPLACE '<BUCKET_ID>' with your GCS BUCKET_ID
BUCKET_ID = 'phdatathonmodel'
# [END setup]


# ---------------------------------------
# 1. Add code to download the data from GCS (in this case, using the publicly hosted data).
# ML Engine will then be able to use the data when training your model.
# ---------------------------------------
# [START download-data]
# Public bucket holding the census data
bucket = storage.Client().bucket('phdatathonmodel')

# Path to the data inside the public bucket
blob = bucket.blob('carddata.data')
# Download the data
blob.download_to_filename('carddata.data')
# [END download-data]


# ---------------------------------------
# This is where your model code would go. Below is an example model using the census dataset.
# ---------------------------------------
# [START define-and-load-data]
# Define the format of your input data including unused columns (These are the columns from the census data files)
COLUMNS = (
    'CardID',               
    'CardType',            
    'MorningWeekdayTrain',    
    'OtherWeekdayTrain',      
    'MorningWeekdayBus',
    'OtherWeekdayBus',
    'MorningWeekdayTram',
    'OtherWeekdayTram',
    'EveningWeekdayTrain',
    'EveningWeekdayBus',
    'EveningWeekdayTram',
    'LunchWeekdayTrain',
    'LunchWeekdayBus',
    'LunchWeekdayTram',
    'WeekendTrain',
    'WeekendBus',
    'WeekendTram',
    'Weekdaytrainstops',
    'Weekendtrainstops',
    'Weekdaybusstops',
    'Weekendbusstops',
    'Weekdaytramstops',
    'Weekendtramstops',
    'Weekdaysused',
    'Weekenddaysused',
)




# Load the training census dataset
with open('./carddata.data', 'r') as train_data:
    raw_training_data = pd.read_csv(train_data, header=None, names=COLUMNS)

# Remove the column we are trying to predict ('income-level') from our features list
# Convert the Dataframe to a lists of lists
train_features = raw_training_data.drop('CardID', axis=1)


# [START create-pipeline]
# Create pipeline to extract the numerical features

# Create the classifier
classifier = KMeans(n_clusters = 10)

# Transform the features and fit them to the classifier
classifier.fit(train_features)

# Create the overall model as a single pipeline
pipeline = Pipeline('classifier', classifier)

# [END create-pipeline]


# ---------------------------------------
# 2. Export and save the model to GCS
# ---------------------------------------
# [START export-to-gcs]
# Export the model to a file
model = 'model.joblib'
joblib.dump(pipeline, model)

# Upload the model to GCS
bucket = storage.Client().bucket(BUCKET_ID)
blob = bucket.blob('{}/{}'.format(
    datetime.datetime.now().strftime('kmeans_%Y%m%d_%H%M%S'),
    model))
blob.upload_from_filename(model)
# [END export-to-gcs]

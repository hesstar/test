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
BUCKET_ID = 'phdatathonmodel'

bucket = storage.Client().bucket('phdatathonmodel')


blob = bucket.blob('carddata.data')

blob.download_to_filename('carddata.data')


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




with open('./carddata.data', 'r') as train_data:
    raw_training_data = pd.read_csv(train_data, header=None, names=COLUMNS)


train_features = raw_training_data.drop('CardID', axis=1)


classifier = KMeans(n_clusters = 10)


classifier.fit(train_features)


pipeline = Pipeline('classifier', classifier)

model = 'model.joblib'
joblib.dump(pipeline, model)

bucket = storage.Client().bucket(BUCKET_ID)
blob = bucket.blob('{}/{}'.format(
    datetime.datetime.now().strftime('trainkmeans_%Y%m%d_%H%M%S'),
    model))
blob.upload_from_filename(model)


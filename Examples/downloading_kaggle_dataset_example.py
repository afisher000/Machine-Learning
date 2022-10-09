# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 15:52:57 2022

@author: afisher
"""

import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

api = KaggleApi()
api.authenticate()

## Competition dataset
# competition = 'titanic' #Url is kaggle.com/c/competition
# files = ['train.csv','test.csv']
# for file in files:
#     api.competition_download_file(competition, file)
    
    
## Stand alone dataset
user_dataset_name = 'dansbecker/basic-data-exploration' # from url
files = ['train.csv','test.csv','sample_submission.csv']
for file in files:
    api.dataset_download_file(user_dataset_name, file)

## Unzip files (if necessary)
# zipfile_name = ''
# with zipfile.ZipFile(zipfile_name, 'r') as zipref:
#     zipref.extractall()
    
    

#!/usr/bin/env python
"""
use the iris data to demonstrate how logging is tied to 
a machine learning model to enable performance monitoring
"""

import time,os,re,csv,sys,uuid,joblib
from datetime import date
import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from logger import update_predict_log, update_train_log
from cslib import fetch_ts, engineer_features
import unittest
from model import *


def train_model(X,y,saved_model):
    """
    function to train model
    """

    ## Perform a train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    ## Specify parameters and model
    params = {'C':1.0,'kernel':'linear','gamma':0.5}
    clf = svm.SVC(**params,probability=True)

    ## fit model on training data
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test,y_pred))

    ## retrain using all data
    clf.fit(X, y)
    print("... saving model: {}".format(saved_model))
    joblib.dump(clf,saved_model)

    
#def update_predict_log(y_pred,y_proba,query,runtime):
#def update_predict_log(y_pred,y_proba,query,runtime):
def update_predict_log(country,target_date,y_pred,y_proba,MODEL_VERSION,runtime):
    
    """
    update predict log file
    """

    ## name the logfile using something that cycles with date (day, month, year)    
    today = date.today()
    #logfile = "example-predict-{}-{}.log".format(today.year, today.month)
    logfile = "predict-log-{}-{}.log".format(today.year, today.month)

    ## write the data to a csv file    
    header = ['unique_id','timestamp','y_pred','y_proba','x_shape','model_version','runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|')
        if write_header:
            writer.writerow(header)

        #to_write = map(str,[uuid.uuid4(),time.time(),y_pred,y_proba,query.shape,MODEL_VERSION,runtime])
        to_write = map(str,[uuid.uuid4(),time.time(),y_pred,y_proba,MODEL_VERSION,runtime])
        writer.writerow(to_write)


def update_train_log(unique_id, dates, rmse, runtime,
                     MODEL_VERSION, MODEL_VERSION_NOTE,test=True):
    """
    update train log file
    """
    ## name the logfile using something that cycles with date (day, month, year)    
    today = date.today()
    #logfile = "example-train-{}-{}.log".format(today.year, today.month)
    logfile = "train-log-{}-{}.log".format(today.year, today.month)

    ## write the data to a csv file    
    # header = ['unique_id','timestamp','y_pred','y_proba','x_shape','model_version','runtime']
    header = ['unique_id','start date', 'end date', 'rmse', 'runtime', 'model_version', 'model_version_note']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    #with open(logfile_train,'b') as csvfile:
    with open(logfile,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|')
        if write_header:
            writer.writerow(header)

        #to_write = map(unique_id,(str(dates[0]),str(dates[-1])),{'rmse':eval_rmse},runtime,
        #to_write = map(unique_id,(str(dates[0]),str(dates[-1])),{'rmse':rmse},runtime,
                     #MODEL_VERSION, MODEL_VERSION_NOTE,test=True)
        to_write = map(str,[unique_id,(str(dates[0]),str(dates[-1])),{'rmse':rmse},runtime,
                     MODEL_VERSION, MODEL_VERSION_NOTE])    
        writer.writerow(to_write)
        
        
        
def predict(query):
    """
    generic function for prediction
    """

    ## start timer for runtime
    time_start = time.time()
    
    ## ensure the model is loaded
    model = joblib.load(saved_model)

    ## output checking
    if len(query.shape) == 1:
        query = query.reshape(1, -1)
    
    ## make prediction and gather data for log entry
    y_pred = model.predict(query)
    y_proba = None
    if 'predict_proba' in dir(model) and model.probability == True:
        y_proba = model.predict_proba(query)
    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update the log file
    update_predict_log(y_pred,y_proba,query,runtime)
    
    return(y_pred)


class TestModels(unittest.TestCase):
    
    def test_model_train(self):
        data_dir = os.path.join(os.path.expanduser('~'),"AI Workflow Last Module AI in Prod","Capstone","Unittest","cs-train")
        model_train(data_dir)
        self.assertTrue(os.path.exists(MODEL_DIR))
    
    def test_load_model(self):
        loadeddata,loadedModel = model_load()
        self.assertTrue(len(loadedModel)) 
        self.assertTrue(len(loadeddata))
        
    def test_predict_model(self):
        country='all'
        year='2018'
        month='01'
        day='05'
        
        result = model_predict(country,year,month,day)
        y_pred = result['y_pred']
        self.assertTrue(result) 



### Run the tests
if __name__ == '__main__':
    unittest.main()
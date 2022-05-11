# -*- coding: utf-8 -*-
"""
Created on Tue May 10 17:24:52 2022

@author: Pragnesh.Prajapati
"""
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\Pragnesh.Prajapati\Downloads\fakenewsclassifier-349718-fb2f2c137734.json'


from google.cloud import storage#pip install --upgrade google-cloud-storage
client = storage.Client()


import mlflow
mlflow.set_tracking_uri("http://34.100.165.250:5000/")


import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, f1_score, accuracy_score
import pickle
import mlflow.sklearn


path = os.getcwd()
Data_path=os.path.join(path,'news.csv')


#SGD:
news = pd.read_csv(Data_path)
news.fillna("", inplace=True)
X = news['text']
y = news['label']

#Splitting the data into train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

mlflow.set_experiment("SGD")
with mlflow.start_run(run_name="SGD"):
    #hyperparameter
    stop_words='english'
    #mlflow.log_param("alpha", alpha)
    mlflow.log_param("Stopword",stop_words)
    from sklearn.linear_model import SGDClassifier
    #Creating a pipeline that first creates bag of words(after applying stopwords) & then applies Multinomial Naive Bayes model
    pipelineSGD = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),('nbmodel', SGDClassifier())])
    
    
    #Training our data
    print(pipelineSGD)
    pipelineSGD.fit(X_train, y_train)
    
    mlflow.sklearn.log_model(pipelineSGD, "model_SGD")
    from google.cloud import storage as gcs_storage#Predicting the label for the test data
    pred = pipelineSGD.predict(X_test)
    
    #Checking the performance of our model
    print(classification_report(y_test, pred, zero_division=1))
    print("Accuracy_SGD:", accuracy_score(y_test, pred))
    print('F1_Score_SGD:', f1_score(y_test, pred, average="weighted"))
    print(confusion_matrix(y_test, pred))
    cm=classification_report(y_test, pred, zero_division=1)
    mlflow.log_metric("AccuracySGD", accuracy_score(y_test, pred))
    mlflow.log_metric('F1ScoreXGBoost', f1_score(y_test, pred, average="weighted"))
mlflow.end_run()
#*******************************************************************#
#SVC Model:
from sklearn.svm import SVC
mlflow.set_experiment("SVC")
with mlflow.start_run(run_name="SVC"):
    mlflow.log_param("Stopword",stop_words)
    #Creating a pipeline that first creates bag of words(after applying stopwords) & then applies Multinomial Naive Bayes model
    pipelineSVC = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),('nbmodel', SVC())])
    
    
    #Training our data
    print(pipelineSVC)
    pipelineSVC.fit(X_train, y_train)
    
    mlflow.sklearn.log_model(pipelineSVC, "model_SVC")
    
    #Predicting the label for the test data
    pred = pipelineSVC.predict(X_test)
    
    #Checking the performance of our model
    print(classification_report(y_test, pred, zero_division=1))
    print("AccuracySVC:", accuracy_score(y_test, pred))
    print('F1_Score_SVC:', f1_score(y_test, pred, average="weighted"))
    print(confusion_matrix(y_test, pred))
    cm=classification_report(y_test, pred,zero_division=1)
    mlflow.log_metric("AccuracySVC", accuracy_score(y_test, pred))
    mlflow.log_metric('F1_ScoreSVC', f1_score(y_test, pred, average="weighted"))
                  
mlflow.end_run()
#*******************************************************************#
#RandomForest:
from sklearn.ensemble import RandomForestClassifier

mlflow.set_experiment("RandomForest")
with mlflow.start_run(run_name="RandomForest"):
    mlflow.log_param("Stopword",stop_words)
    n_estimators=10
    max_depth=100
    min_samples_leaf=2
    mlflow.log_param('nestimators', n_estimators)
    mlflow.log_param('maxdepth', max_depth)
    mlflow.log_param('minsamplesleaf', min_samples_leaf)    
    # Create the model with 100 trees
    RForest = RandomForestClassifier(criterion='entropy',n_estimators=10,max_depth=100,bootstrap = True,max_features = 'sqrt',min_samples_leaf=2,random_state=123,)
    
    
    pipelineRF = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),('nbmodel', RForest)])
    
    
    #Training our data
    print(pipelineRF)
    pipelineRF.fit(X_train, y_train)
    
    mlflow.sklearn.log_model(pipelineRF, "model_RF")
    #Predicting the label for the test data
    pred = pipelineRF.predict(X_test)
    
    #Checking the performance of our model
    print(classification_report(y_test, pred, zero_division=1))
    print("AccuracyRF:", accuracy_score(y_test, pred))
    print('F1_Score_RF:', f1_score(y_test, pred, average="weighted"))
    print(confusion_matrix(y_test, pred))
    mlflow.log_metric("AccuracyRF", accuracy_score(y_test, pred))
    mlflow.log_metric('F1ScoreRF', f1_score(y_test, pred, average="weighted"))
mlflow.end_run()
#**************************************************************************
#***************************************************************************8
#DeciionTree
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

mlflow.set_experiment("DeciionTree")
with mlflow.start_run(run_name="DeciionTree"):
    criterion='entropy'
    mlflow.log_param("Stopword",stop_words)
    mlflow.log_param('criterion',criterion)
    DTree = DecisionTreeClassifier(criterion='entropy',random_state=0)
    print(DTree)
    
    pipelineDT = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),('nbmodel', DTree)])
    
    
    #Training our data
    print(pipelineDT)
    pipelineDT.fit(X_train, y_train)
    
    mlflow.sklearn.log_model(pipelineDT, "model_DTree")
    #Predicting the label for the test data
    pred = pipelineDT.predict(X_test)
    
    #Checking the performance of our model
    print(classification_report(y_test, pred, zero_division=1))
    print("Accuracy_DT:", accuracy_score(y_test, pred))
    print('F1_Score_DT:', f1_score(y_test, pred, average="weighted"))
    print(confusion_matrix(y_test, pred))
    mlflow.log_metric("AccuracyDTree", accuracy_score(y_test, pred))
    mlflow.log_metric('F1ScoreDTree', f1_score(y_test, pred, average="weighted"))
mlflow.end_run()    
#*******************************************************************#
#XGBOOST:
from xgboost import XGBClassifier

news = pd.read_csv(Data_path)
news.fillna("", inplace=True)

#converting target values into  numerical format:
#Not using label encode----> Having other inputs also apart from REAL and Fake    
news['label']= news.label.apply(lambda x: 0 if x=='REAL' else 1 ) 
X = news['text']
y = news['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
mlflow.set_experiment("Xgboost")
with mlflow.start_run(run_name="Xgboost"):
    max_depth =6
    min_child_weight=1
    eta=.3
    subsample=1
    mlflow.log_param("Stopword",stop_words)
    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('min_child_weight', min_child_weight)
    mlflow.log_param('eta',eta)
    mlflow.log_param('subsample', subsample)
             
    
    pipelineXGB = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),('nbmodel', XGBClassifier())])
    #Training our data
    
    pipelineXGB.fit(X_train, y_train)
    
    
    mlflow.sklearn.log_model(pipelineXGB, "model_XGB")
    #Predicting the label for the test data
    pred = pipelineXGB.predict(X_test, )
    
    def get_newsType(newstype):
        type = []
        for i in newstype:
            if i == 0:
                type.append("REAL")        
            else:
                type.append("FAKE")
        return type
    pred = get_newsType(pred)
    y_test = get_newsType(y_test)
    
    #Checking the performance of our model
    print(classification_report(y_test, pred, zero_division=1))
    print("Accuracy_XGB:", accuracy_score(y_test, pred))
    print('F1_Score_XGB:', f1_score(y_test, pred, average="weighted"))
    print(confusion_matrix(y_test, pred))
    mlflow.log_metric("AccuracyXGBoost",accuracy_score(y_test, pred))
    mlflow.log_metric('F1ScoreXGBoost', f1_score(y_test, pred, average="weighted"))
    
    filename = (r'Trained_XGB_Model.sav')
    pickle.dump(pipelineXGB, open(filename, 'wb'))
mlflow.end_run()

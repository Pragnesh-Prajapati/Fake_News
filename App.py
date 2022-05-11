
from pydoc import render_doc  
#from aem import app
from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from flask_cors import CORS #pip install flask_cors
import os
import newspaper # pip install newspaper3k
from newspaper import Article
import urllib
import nltk
nltk.download('punkt')

app=Flask(__name__) # Initialise flask app
CORS(app)


 # Launch app by rendering/using html page
@app.route('/',methods=['GET'])
def HTML_Dev():
    return render_template('App_HTML_template.html')


 # Logic for the form on html page. Import data from html to python

@app.route('/',methods=['GET','POST'])
def predict():
    url =request.get_data(as_text=True)[5:]
    url = urllib.parse.unquote(url)
    article = Article(str(url))
    article.download()
    article.parse()  # command line arguments
    article.nlp()
    news = article.summary
    # Passing the news article to the model and returing whether it is Fake or Real
    loaded_model = pickle.load(open((r'Trained_Model.sav'), 'rb'))  # Load the frozen model
    pred = loaded_model.predict([news])
    return render_template('App_HTML_template.html', prediction_text='Provided news articles is ---> "{}"'.format(pred))     # Output display on the html page

 # Run the app via a port
if __name__=='__main__':
    app.run(port=3000,debug=True)

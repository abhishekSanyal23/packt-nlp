#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 14:47:05 2018
Packt A sample text classifier
@author: abhishek
"""
from nltk.corpus import reuters
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#Setting up the stop words as provided by nltk corpus
sw = set(stopwords.words('english'))

#All categories in nltk corpus
#print (reuters.categories())

#We are only going to take sample rubber and cotton
Xr = np.array(reuters.sents(categories=['rubber']))
Xc = np.array(reuters.sents(categories=['cotton']))
Xw = np.concatenate((Xr,Xc))

# Preprocessing the X axis - .strip() 
# removes all whitespace at the start and end, 
# including spaces, tabs, newlines and carriage returns
# .lower() to lower case, so it ignores case while traiing the doc.
X = []
for document in Xw:
    X.append(' '.join(document).strip().lower())
    
#Setting up the Y axis (or output)
# Cotton - 1, Rubber - 0
Yr = np.zeros(shape = Xr.shape)
Yc = np.ones(shape = Xc.shape)
Y = np.concatenate((Yr,Yc))

#defining tokenizer - as learnt in the bag of words strategy
#Vectorizing process
ret = RegexpTokenizer('[a-zA-Z0-9\']+')
sw = set(stopwords.words('english'))
ess = SnowballStemmer(language='english', ignore_stopwords=True)
def tokenizer(sentence):
    tokens = ret.tokenize(sentence)
    return [ess.stem(t) for t in tokens if t not in sw]

#Using tf-idf
tfidf = TfidfVectorizer(tokenizer=tokenizer, ngram_range=(1,2), norm='l2')
Xv = tfidf.fit_transform(X)

#Using Random Forest Classifier
#Train Test split

X_train, X_test, Y_train, Y_test = train_test_split(Xv, Y, test_size = 0.25)
rf = RandomForestClassifier(n_estimators=25)
rf.fit(X_train, Y_train)
score = rf.score(X_test, Y_test)
print ('Score - %.3f'% score)

#Getting a score of Score - 0.898

#Using the trained model to predict now
category = ['Rubber', 'Cotton']
test_newsline = ['Tight supply to keep cotton prices in India firm in FY19']
yvt = tfidf.transform(test_newsline)
out_cat = int(rf.predict(yvt)[0])
print ('Predicted category : ' + category[out_cat])


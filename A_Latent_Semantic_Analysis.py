#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 7 14:47:05 2018
Packt A Latent semantic analysis
Latent Semantic Analysis (LSA)
Probabilistic LSA (PLSA)
Latent Dirichlet Allocation
@author: abhishek
"""


#LSA
from nltk.corpus import brown
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.linalg import svd
import numpy as np

#Take news section
sentences = brown.sents(categories = ['news'])[0:500]
corpus = []
for sentence in sentences:
    corpus.append(' '.join(sentence))
    
#Vectorize
vectorizer = TfidfVectorizer(strip_accents='unicode', stop_words='english', norm='l2', sublinear_tf=True)
Xc = vectorizer.fit_transform(corpus).todense()


# To understand the section watch the video -
# LSA_How_it_happens.webm
# Source - wikipedia
# Animation of the topic detection process in a document-word matrix. 
# Every column corresponds to a document, every row to a word. 
#A cell stores the weighting of a word in a document (e.g. by tf-idf), dark cells 
# indicate high weights. LSA groups both documents that contain similar words, 
# as well as words that occur in a similar set of documents. 
# The resulting patterns are used to detect latent components.

U, s, v = svd(Xc, full_matrices=False)


rank = 10
Uk = U[:, 0:rank]
sk = np.diag(s)[0:rank, 0:rank]
Vk = v[0:rank, :]


Mtwks = np.argsort(Vk, axis=1)[::-1]
for t in range(rank):
    print ('\nTopic' + str(t))
    for i in range(10):
        print(vectorizer.get_feature_names()[Mtwks[t,i]])
        

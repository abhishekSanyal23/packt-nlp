#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 14:47:05 2018
Packt Bag of words strategy
@author: abhishek
"""


#Sentence Tokenization
from nltk.tokenize import sent_tokenize

generic_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, \
                sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. \
                Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris \
                nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in \
                reprehenderit in voluptate velit esse cillum dolore eu fugiat \
                nulla pariatur. Excepteur sint occaecat cupidatat non proident, \
                sunt in culpa qui officia deserunt mollit anim id est laborum."

print (sent_tokenize(generic_text))


english_text = "Where is the nearest train station? I need to go to London."

print (sent_tokenize(english_text, language='english'))


spanish_text = "¿Dónde está la estación de tren más cercana? Necesito ir a Londres."

print (sent_tokenize(spanish_text, language='spanish'))

#Word Tokenize
from nltk.tokenize import TreebankWordTokenizer

tbwt = TreebankWordTokenizer()

print (tbwt.tokenize(english_text))

complex_text = "This is a free country, isn't it?"

print (tbwt.tokenize(complex_text))


#Reg Exp tokenize
from nltk.tokenize import RegexpTokenizer

reg = RegexpTokenizer("[a-zA-Z0-9\']+")

print (reg.tokenize(complex_text))

#Stop words removal
from nltk.corpus import stopwords

sw = set(stopwords.words('english'))

print (sw)

out = [word.lower() for word in reg.tokenize(complex_text) if word.lower() not in sw]
print (out)

#Stemming

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer(language='english', ignore_stopwords=True)
print (stemmer.stem('running'))

from nltk.stem import LancasterStemmer
stemmer = LancasterStemmer()
print (stemmer.stem('running'))

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
print (stemmer.stem('running'))

#Vectorizing

from sklearn.feature_extraction.text import CountVectorizer
corpus = [
        'This is a simple corpus',
        'A corpus is a set of documents',
        'We want to analyze the corpus and the documents',
        'Documents can be automatically tokenized'
        ]

cv = CountVectorizer()
vectorized_corpus = cv.fit_transform(corpus)
print (vectorized_corpus.todense())

print (cv.vocabulary_)
vector = [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0]
print (cv.inverse_transform(vector))

#Vectorizing process
ret = RegexpTokenizer('[a-zA-Z0-9\']+')
sw = set(stopwords.words('english'))
ess = SnowballStemmer(language='english', ignore_stopwords=True)
def tokenizer(sentence):
    tokens = ret.tokenize(sentence)
    return [ess.stem(t) for t in tokens if t not in sw]

cv = CountVectorizer(tokenizer=tokenizer)
vectorized_corpus = cv.fit_transform(corpus)
print(vectorized_corpus.todense())

vector = [0, 1, 0, 1, 0, 0, 1, 0]
print (cv.inverse_transform(vector))

cv = CountVectorizer(tokenizer=tokenizer, ngram_range=(1,2))
vectorized_corpus = cv.fit_transform(corpus)
print(vectorized_corpus.todense())

vector = [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0]
print (cv.inverse_transform(vector))
print (cv.vocabulary_)


#tf-idf 
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
vectorized_corpus = tfidf.fit_transform(corpus)
print (vectorized_corpus.todense())

print (tfidf.vocabulary_)


tfidf = TfidfVectorizer(tokenizer=tokenizer, ngram_range=(1,2), norm='l2')
vectorized_corpus = tfidf.fit_transform(corpus)
print (vectorized_corpus.todense())

print (tfidf.vocabulary_)
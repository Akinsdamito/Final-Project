# import libraries
import os
import json
import nltk
import re
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, GRU
from tensorflow.keras.layers import Dropout, Activation, Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Sequential
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from gensim.test.utils import datapath
from gensim import utils
from nltk.tokenize import word_tokenize
import sqlite3
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import time  # To time our operations
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_score, recall_score, make_scorer, f1_score, accuracy_score,hamming_loss
import pkg_resources
from symspellpy.symspellpy import SymSpell
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import spacy
from wordcloud import WordCloud
from langdetect import detect
from gensim.test.utils import datapath
from gensim import utils
from gensim.utils import simple_preprocess
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

def make_word_cloud(df,column,category, 
                    stopwords=stopwords.words('english')):
    """
        This function takes in dataframe, columns name as a string, type of news headline category,
        and make a word cloud plot of the news category
        
        
        return: word cloud plot of the news category
    """
    
    data=df[df[column]==category]['headline']
    text = " ".join(review for review in data)
    cloud = WordCloud(max_words=100,
                            width=2000, 
                            height=1100, 
                            stopwords=stopwords).generate(text)
    return cloud

#Word count
def word_count(text):
  
    """
        This function takes in text as a string, and separate it into different words
        
        
        return: each words with their counts 
    """
    c_vec = CountVectorizer(tokenizer=None)
    ngrams = c_vec.fit_transform(text)

    count_values = ngrams.toarray().sum(axis=0)
 
    vocab = c_vec.vocabulary_
 
    df_unigram = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
            ).rename(columns={0: 'frequency', 1:'Unigram'})
    return df_unigram


def setence_segmentation(text):

    """
        This function takes in text as a string, and breaks words that are joined together into different words.
        E.g ‘edmontondeliveredcreated' will be return as ‘edmonton, delivered, created'
        
        
        return: modified text 
    """
  
  # Set max_dictionary_edit_distance to zero so as to avoid spelling correction
  
    sym_spell = SymSpell(max_dictionary_edit_distance=0, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
  
    # term_index is the column of the term and count_index is the
    # column of the term frequency
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

      # a sentence without any spaces
    seg_sent=[]
    for sent in text:
        try:
            input_term = sent
            result = sym_spell.word_segmentation(input_term)
            new_sentence=(result.corrected_string).replace('c ovid','covid')
            seg_sent.append(new_sentence)
        except IndexError:
            seg_sent.append(input_term)
            pass
    return seg_sent

def clean_text_process(text,stopwords):
    """
        This function takes in text as a string,array of additional words in the text if we want to remove
        them. It replace symbols with space and also remove the stopwords if Location is None, but if
        is not None, at add the array to the Stopwords be perfoming the removal of words.
        
        return: modified text
    """
    #for word in text:
    replace_symbol = re.compile('[/(){}\[\]\|@,;?:\-\.]')
    final_text=[]    
    for  i in text:  
#        print(i)
    # lowercase text    
        text = i.lower()
    # Single character removal
        text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)

    # Removing multiple spaces
        text = re.sub(r'\s+', ' ', text)  
      
    # replace replace_symbol symbols by space in text.
        text = replace_symbol.sub(' ',text) 

    # remove symbols which are not in [a-zA-Z_0-9] from text
        text = re.sub(r'\W+', ' ', text)
    
    # remove symbols which are not in numeric from text
        text = re.sub(r'\d', ' ', text)
              
    # remove numbers from text
        text = re.sub('[0-9]', ' ', text)
    #STOPWORDS = stopwords.words('english')
        
        text = ' '.join(word for word in text.split() if word not in stopwords)
            
        final_text.append(text)
    return final_text
  

def get_all_person(text):
    
    """
        This function takes in text as a string, and identify all the name
        stopwords in the text.
        
        return: An array of Names.
    """
    nlp_lg= spacy.load("en_core_web_lg")
    nlp_en = spacy.load('en_core_web_sm')
    names = []

    for fn in text:
        doc = nlp_lg(fn)
        names.extend([[fn, ent.text, ent.start, ent.end] for ent in doc.ents if ent.label_ in ['PERSON']])
 
    df = pd.DataFrame(names, columns=['File', 'names', 'start','end'])
    return df['names'].values
  
def get_all_location(text):
    
    """
        This function takes in text as a string, and identify all the location
        stopwords in the text.
        
        return: An array of Locations.
    """
    nlp_lg = spacy.load("en_core_web_lg")
    nlp_en = spacy.load('en_core_web_sm')
    locations = []

    for fn in text:
        doc = nlp_lg(fn)
        locations.extend([[fn, ent.text, ent.start, ent.end] for ent in doc.ents if ent.label_ in ['GPE']])
 
    df = pd.DataFrame(locations, columns=['File', 'Location', 'start','end'])
    return df['Location'].values
  

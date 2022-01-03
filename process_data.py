# import libraries
import os
import sys
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
from python_functions import *
from langdetect import detect


spacy.load('en_core_web_sm')

spacy.load("en_core_web_lg")


def load_data(data_filepath):
    
    """
        Description -This function takes in the files path and read it into dataframe
        
        
        return: It returns a dataframe. 
   """
    
    df = pd.read_json(data_filepath, lines=True)
    
    return df

def drop_na_duplicate(df):
    """
    Description- This fuction takes in a dataframe and remove the missing and duplicate
    in column headline'
    Output- return a dataframe
    
    """
    df = df[df['headline'] != '']
    new_df=df.drop_duplicates(subset=['headline'])
    return new_df

def drop_non_english(df):
    """
    Description- This fuction takes in a dataframe and remove the non english
    text in column headline'
    Output- return a dataframe with a new column language
    
    """
    df['language'] = df['headline'].apply(detect)
    new_df=df[df['language']=='en']
    return new_df

def save_data(df,label_column,clean_narative,data_filename):
    
    """
        Description -This function save the data into a database
        
        
         
   """
    
    model_df=pd.DataFrame(data=clean_narative,columns=['clean_headline'])
    model_df[label_column]=df[label_column]
    model_df['sent_length']=model_df['clean_headline'].map(lambda x: len(x))
    model_df.to_csv(data_filename,index=False)


def main():
    if len(sys.argv) == 3:

        data_filepath, modeldata_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES:{}'
              .format(data_filepath))
        df = load_data(data_filepath)
        
        # removing duplicate, missing values and non english headlines
        print("removing duplicate, missing values and non english headlines starts ......")
        start = time.time()
        new_df = drop_non_english(drop_na_duplicate(df)).reset_index(drop=True)
        print("removal completed.: {} mins".format(round((time.time()-start)/60 , 2)))
        
        data_set=new_df['headline'].tolist()
        
        ## List of Stopwords in NLTK library
        STOPWORDS = stopwords.words('english')
        
        ## Adding our own stopwords
        new_stword = pd.read_csv('newstopword', sep=",", header=0)['Unigram'].tolist()
        STOPWORDS.extend(new_stword)
        
        # Text segmentation done on the headline column, i.e words that are join together are separeted
        print("Corpus's segmentation starts ......")
        start = time.time()
        new_narative= setence_segmentation(data_set)
        print("Corpus's segmentation completed.: {} mins".format(round((time.time()-start)/60 , 2)))

        ## De-noising the dataset and normalisation
        print("Corpus's De-noising and normalisation starts ......")
        start = time.time()
        clean_da=clean_text_process(new_narative,stopwords=STOPWORDS)
        
        print("Corpus's De-noising the dataset and normalisation completed.: {} mins".format(round((time.time()-start)/60 , 2)))
        
        print('Saving data...\n    DATABASE: {}'.format(data_filepath))
        save_data(df,'category',clean_da,modeldata_filepath)
        
        print('Cleaned data saved to CSV!')
    
    else:
        print('Please provide the filepaths of the dataset '\
              'well as the filepath of the CSV to save the cleaned data '\
              'to as the second argument. \n\nExample: python process_data.py '\
              'Analysis_data.json'\
              'model_df.csv')

        

if __name__ == '__main__':
    main()    






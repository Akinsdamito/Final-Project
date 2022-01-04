# import libraries
import nltk
import re
import sys
import sqlite3
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import gc 
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, make_scorer, f1_score, accuracy_score,hamming_loss
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from skmultilearn.adapt import MLkNN
from skmultilearn.problem_transform import ClassifierChain
from nltk.corpus import wordnet
import time
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
from python_functions import *

def load_data(data_filepath):
    
    """
        This function takes in the database path and it reads the data
        
        
        return: The messages which are the predators, the response i.e the labels as a dataframe 
            and an array of all the labels
    """
    df=pd.read_csv(data_filepath, sep=",", header=0)
    new_df=df[df['sent_length']>20].dropna(subset=['clean_headline']).reset_index(drop=True)
    
    return new_df 

def preprocess(data_set,STOPWORDS):
    
    """
        This function takes in text and stopwords, it uses the function WORDCOUNT to count the 
        occurence of each word and add word with only one occurence to the stopwords, then uses function 
        CLEAN_DATA to remove the combined stopwords and also preprocesed the data
        
        return: Text devoid of noise.
    """
    # Count of each tokens in the dataset
    start = time.time()
    print("getting less frequent words in dataset ......")
    wordcount=word_count(data_set)
    new_stopword=wordcount[wordcount['frequency']==0]['Unigram'].values.tolist()
    print('collection of words completed.: {} mins'.format(round((time.time()-start)/60 , 2)))
    ## Adding our own stopwords
    STOPWORDS.extend(new_stopword)

    ## De-noising the dataset and normalisation
    print("starting data preprocessing ......")
    clean_data=clean_text_process(data_set,stopwords=STOPWORDS)
    print('data preprocessing completed.: {} mins'.format(round((time.time()-start)/60 , 2)))

    return clean_data


def tokenize(text):
    
    """
        This function seperated the text to different tokens
        
        return: Tokens.
    """
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in text:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    tokens = word_tokenize(' '.join(c for c in clean_tokens ))
    
    
    return tokens

def build_model():
    
    """
        This function build the model pipeline by using grid search to get
        the best parameters.
        
        return: Pipeline.
    """
    
    pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(sublinear_tf=True,norm='l2',
                        ngram_range=(1, 2),)),
                              
                ('mnb', MultinomialNB())
    ])

    # specify parameters for grid search
    parameters = {
    'tfidf__min_df': np.array([10, 20,30,40,50]), 
    'tfidf__sublinear_tf': np.array([True,False]),
    'mnb__alpha': np.linspace(0.1, 1.5, 10),
               
        
    }

    # create grid search object
    cv =  GridSearchCV(pipeline, param_grid=parameters, scoring='accuracy')
    
       
    return cv

def evaluate_model(model, X_test, 
                   Y_test, category_names):
    """
        This function evaluate the model 
        
        return: The classification reports for each label.
    """

    prediction = model.predict(X_test)
    
    print('\t\t\t\tCLASSIFICATIION METRICS\n')
    print(metrics.classification_report(Y_test, prediction, 
                                    target_names= category_names))
        
def save_model(model, model_filepath):
    
    """
        This function save the model as a pickle file
        
        return: The classification reports for each label.
    """
    filename =  model_filepath
    pickle.dump(model, open(filename, 'wb')) 
    
STOPWORDS = stopwords.words('english')

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        df=load_data(database_filepath)
        df['new_cln_data']=preprocess(df['clean_headline'],STOPWORDS)
        category_names=df['category'].unique()
        
        ## Changing label to categorical values
        news_labels=df['category'].unique()
        news_labels_dict={}
        for index in range(len(news_labels)):
            news_labels_dict[news_labels[index]]=index
        labels = df['category'].apply(lambda x: news_labels_dict[x])
       
    # Splitting to training and test split.

        X_train, X_test, y_train, y_test = train_test_split(df['new_cln_data'],labels, 
                                                               test_size=0.25, 
                                                               random_state=3)
        
        
        print('Building model...')
        model = build_model()
        
        try:
            print('Training model...')
            start = time.time()
            model.fit(X_train, y_train)
            print('Training model.: {} mins'.format(round((time.time()-start)/60 , 2)))
            
        except RuntimeWarning:
            pass
        
        print('Evaluating model...')
        evaluate_model(model, X_test,  y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()    
# Final-Project
In this project, an NLP model is trained to classify new headlines into four different categories. The dataset used was explored and cleaned up by performing an ETL process.

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project and goal">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#Exploratory data analysis">Exploratory data analysis</a></li>
      </ul>
      <ul>
        <li><a href="#Text Preprocessing">Text Preprocessing</a></li>
      </ul>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
      </ul>
    </li>
    <li><a href="#Uploded Files">Uploded Files</a></li>    
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

# About-the-project
This involes data exploration, cleaning, transformation, loading of clean data into an csv file and finally training of an NLP model for classification.
The data used is obtained from kaggle website. The data consist of two hundred thousand of nodes consisting of 4 columns which include news headlings and their categories.
The objective of this work is to tran a classification model, that can be use to classify news headlines to different categories. The original dataset has 41 categories, and the dataset is large. To be able to run the model on my machine, I randomly select 25000 nodes out of the entire dataset, and also reduce the labels to 4 category.
The project is divided into 3 parts:
* The first part of the project, I perform data exploratory to have an idea how the data look. Check for dublication in the dataset and visualizing the data. 
* The second part performed an ETL process on two data datasets i.e dropping rows with missing data, removing headlines not written in eglish language, carrying out word segmentation i.e separating words that are joined together into different words. The clean data set is load into a csv file.
* Lastly, I created an NLP pipeline and Machine learning pipeline process. Since it is impossible to train a model with a text data, so we converted the messages to a word of vector before training the machine learning model. Two approaches were used to convert the messages to vectors, I used CountVectorizer and TfidfTransformer when creating non neural network model, and used Word2Vec word embedding approach when training a neural network model using Keras.
# Getting Started
This project codes are written  in python and group into 3:
## Exploratory data analysis
The data set consist of 6 columns, which are the category, headline, authors, link, short description and date, but I only focus on 2 of the columns which are category and headline in traning the models. The code for the data exploration is written in a notebook, and is uploaded as a file name dataexplore.ipynb.
## Text Preprocessing
Differents steps were carried out in processing the text; 
* Duplicate data are removed and missing values are dropped.
* All non-English language headlines dropped.
* All punctuations signs, numbers, and unnecessary symbols were removed, and all alphabets were converted to lower case.
* Word segmentation was done to separate words that are joined together e.g 'edmontondeliveredcreated' will be returned as 'edmonton, delivered, created'.
* De-noising the dataset, normalization, and removal of the stopwords.
The code for the data preprocessing is written in the form of a script, and it is save as process_data.py. To run it in a note book you use this command ' %run -i process_data.py Analysis_data.json model_df.csv'. Where process_data.py is the name of the script file, Analysis_data.json is the name of the data file and model_df.csv is the output file, which is contain a clean dataset.
## Model Training
Before training the model, I first did word Lemmatization before converting the text to vector since maching learning model can not be train using text. Two approaches are used for text vectorization; the first is the use of TFIDFVectorize from Sklearn, and the other is the use Word2Vec word- embeddings approach.


## Pre-requisites
In order to run the code successfully, the following liberaries have to be installed on your notebook
* Pandas, latest version
* Scikit-Multilearn
* Keras
* NLTK
* Gensim
* Seaborn
* mathplotlib
* Numpy
* Sklearn
* Langdetect
* Wordcloud
* Spacy
* Sysspellpy

# Uploded Files
The following file are uploded in this repository:
* Code for data exploration
* The code for the extract, transformation and loading the data into CSV file, which is written in python.
* The data files used for the analysis.
* Codes for the training the models using non neural network and neural network approach
* A readme file that explains the information in the data.


# Acknowledgements
* This project wouldn't be possible if not for kaggle team, who provided the data used .


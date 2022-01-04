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
        <li><a href="#Model Training">Model Training</a></li>
      </ul>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
      </ul>
    </li>
    <li><a href="#Uploaded Files">Uploded Files</a></li>    
     <li><a href="#Conclusion">Conclusion</a></li> 
    <li><a href="#acknowledgments">Acknowledgements</a></li>
  </ol>
</details>

# About-the-project
The project involves data exploration, cleaning, transformation, loading of clean data into a CSV file, and finally training of an NLP model for classification. The data used is obtained from Kaggle website. The data consist of two hundred thousand of nodes consisting of 4 columns which include news headlines and their categories. The objective of this work is to train a classification model, that can be used to classify news headlines into different categories. The original dataset has 41 categories, and the dataset is large. To be able to run the model on my machine, I randomly select 25000 nodes out of the entire dataset, and also reduce the labels to 4 categories. The project is divided into 3 parts:
* In the first part of the project, I perform data exploratory to have an idea of how the data look. Check for duplication in the dataset and visualize the data
* The second part performed an ETL process on two data datasets i.e dropping rows with missing data, removing headlines not written in the English language, carrying out word segmentation i.e separating words that are joined together into different words. The clean data set is loaded into a CSV file.
* Lastly, I created an NLP pipeline and a Machine learning pipeline process. Since it is impossible to train a model with text data, so we converted the messages to a word of vector before training the machine learning model. Two approaches were used to convert the messages to vectors, I used CountVectorizer and TfidfTransformer when creating a non-neural network model, and used the Word2Vec word embedding approach when training a neural network model using Keras.
# Getting Started
These project codes are written in python and group into 3:
## A. Exploratory data analysis
The data set consists of 6 columns, which are the category, headline, authors, link, short description, and date, but I only focus on 2 of the columns which are category and headline in training the models. The code for the data exploration is written in a notebook and is uploaded as a file name dataexplore.ipynb.
## B. Text Preprocessing
Differents steps were carried out in processing the text; 
* Duplicate data are removed and missing values are dropped.
* All non-English language headlines dropped.
* All punctuations signs, numbers, and unnecessary symbols were removed, and all alphabets were converted to lower case.
* Word segmentation was done to separate words that are joined together e.g 'edmontondeliveredcreated' will be returned as 'edmonton, delivered, created'.
* De-noising the dataset, normalization, and removal of the stopwords. The code for the data preprocessing is written in the form of a script, and it is saved as process_data.py. To run it in a notebook you use this command ' %run -i process_data.py Analysis_data.json model_df.csv'. Where process_data.py is the name of the script file, Analysis_data.json is the name of the data file and model_df.csv is the output file, which contains a clean dataset.
## C. Model Training
Before training the model, Lemmatization was done followed by conversion of the text to vector since the machine learning model can not be trained using text. Two approaches are used for text vectorization; the first is the use of TFIDFVectorize from Sklearn when training the model with a non-neural network algorithm, and the other is the use Word2Vec word- embeddings approach when training neural network model. The metric used in accessing the model's skill is the accuracy of the model classification. The codes for the non-neural network are written in a notebook that contained explanations and the different methods used, the node book is in the directory name NonneuralNetwork, also for the neural network model, the notebook is in a directory named Neural_Network folder. The final model code is written in the form of a script, and it is saved as NonNeural_train.py. To run it in a notebook you use this command ' %run -i NonNeural_train.py model_df.csv classifierbb.pkl'. Where NonNeural_train.py is the name of the script file, model_df.csv is the name of the data file, and classifierbb.pkl is the output model, which is the trained model saved as a pickle file.
## Pre-requisites
In order to run the code successfully, the following libraries have to be installed on your notebook
* Pandas, the latest version
* Scikit-Multilearn
* Keras
* NLTK
* Gensim
* Seaborn
* Mathplotlib
* Numpy
* Sklearn
* Langdetect
* Wordcloud
* Spacy
* Sysspellpy

# Uploaded Files
The following files are uploaded to this repository:
* Code for data exploration is written in a notebook, and it is named dataexplore.ipynb
* The data files used for the analysis named Analysis_data.json
* The code for the extract, transformation, and loading of the data into CSV file, which is written in python and file name process_data.py.
* Python functions are used in the project and the file name is python_functions.py .
* Codes for the training the models using non-neural network and neural network approach, the directories for the two approaches are Neural_Network and NonneuralNetwork
* Python script to train the final model save in file named NonNeural_train.py.
* The file containing additional stopwords to be removed from the dataset file name newstopword
* A readme file that explains the information in the data.

# Conclusion
A classification model is constructed using different algorithms, but the algorithm that gave the best result is the MultinomialNB. But the model accuracy is less than 50%, which implies that there is room for improvement. One way to improve the model is to do more in cleaning the data set, we can also identify the part of speech in each of the headlines, and apply it when training the model. Another way is to use other embedding algorithms such as BERT, Glove instead of using the ones I used.

# Acknowledgments
* This project wouldn't be possible if not for Kaggle team, who provided the data used.


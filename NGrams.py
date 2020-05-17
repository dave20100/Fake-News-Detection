import pandas as pd
from enum import Enum
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import copy
import time
import nltk
import re
from nltk.corpus import stopwords

#Set a range of Ngram length that will be tested
NgramSizeRange = range(1,3)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# print(stopwords)
datasplit = {
    "char": [],
    "word": []
}

extractedData = {
    "bagOfWords": copy.deepcopy(datasplit),
    "tfidf": copy.deepcopy(datasplit)
}

#Prepares data by removing not needed columns and encoding labels
def basicPreparation(fileName): 
    file = pd.read_csv(fileName)
    label_encoder = preprocessing.LabelEncoder()
    file['label'] = label_encoder.fit_transform(file['label']) #labelling data FAKE = 0 REAL = 1
    file = file.applymap(lambda s: s.lower() if type(s) == str else s) #set everything to lowercase
    return (file['text'], file['label'])

def removeSpecialCharacters(character):
    if(character == '.'):
        return ' '
    if(character.isalnum() or character == ' '):
        return character
    return ''


def dataPreprocessing(articles, labels):
    deletionIndexes = []
    for articleIndex in range(len(articles)):
        articles[articleIndex] = ''.join(w+' ' for w in articles[articleIndex].split(' ') if not w in stop_words and w != '')  # remove special characters
        articles[articleIndex] = re.sub(r'[^a-zA-Z]+', ' ', articles[articleIndex])
        articleLength = len(articles[articleIndex])
        if(articleLength == 0 or articleLength < 500 or articleLength > 5000):
            deletionIndexes.append(articleIndex) #save indexes of rows that need to be deleted
    articles = articles.drop(deletionIndexes, axis=0)
    labels = labels.drop(deletionIndexes, axis=0)
    return (articles, labels)

def plotData():
    return

#Prepare data
(text, labels) = basicPreparation("news.csv") 
(text, labels) = dataPreprocessing(text, labels)

# Extract information from data


for size in NgramSizeRange:
    extractedData["bagOfWords"]["char"].append({"vectorizer": CountVectorizer(analyzer="char", ngram_range=(size, size))})
    extractedData["tfidf"]["char"].append({"vectorizer": TfidfVectorizer(analyzer="char", ngram_range=(size, size))})

extractedData["bagOfWords"]["word"].append({"vectorizer": CountVectorizer(
    analyzer="word", ngram_range=(1, 1))})
extractedData["tfidf"]["word"].append({"vectorizer": TfidfVectorizer(
    analyzer="word", ngram_range=(1, 1))})

print('{:>18}  {:>18}  {:>18} {:>18} {:>18} {:>22}'.format(
    "method", "wordType", "size", "classifier", "score", "time"))
for method in extractedData.keys():
    for wordType in extractedData[method].keys():
        for size in range(len(extractedData[method][wordType])):
            extractedData[method][wordType][size]["classificator"] = {
                "SVC": svm.SVC(),
                "KNN": KNeighborsClassifier(n_neighbors=7),
                "RandomForest": RandomForestClassifier(),
                "MLP": MLPClassifier(max_iter=1000)
            }
            for classifier in extractedData[method][wordType][size]["classificator"].keys():
                extractedData[method][wordType][size]["features"] = extractedData[method][wordType][size]["vectorizer"].fit_transform(text)
                # print(extractedData[method][wordType][size]["vectorizer"].get_feature_names())
                X_train, X_test, y_train, y_test = train_test_split(extractedData[method][wordType][size]["features"], labels, test_size=0.60, random_state=42)
                start = time.time()
                extractedData[method][wordType][size]["classificator"][classifier].fit(X_train, y_train)
                end = time.time() 
                print('{:>18}  {:>18}  {:>18} {:>18} {:>18} {:>22}'.format(method, wordType, size+1, classifier,
                        str(round(extractedData[method][wordType][size]["classificator"][classifier].score(X_test, y_test), 2)) + "%",
                        str(round(end-start, 2)) + "s czas uczenia"))


#Save trained models

#Plot data

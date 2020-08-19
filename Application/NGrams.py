from sklearn.decomposition import PCA
import copy
import re
import string
import time
from enum import Enum
import csv
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn import preprocessing, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import (CountVectorizer,
                                             HashingVectorizer,
                                             TfidfVectorizer)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from datetime import datetime

#Set a range of Ngram length that will be tested
minNgram = 1
maxNgram = 6
NgramSizeRange = range(minNgram, maxNgram)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

datasplit = {
    "Ngram": [],
    "Word": []
}

extractedData = {
    "bagOfWords": copy.deepcopy(datasplit),
    "tfidf": copy.deepcopy(datasplit)
}

#Prepares data by removing not needed columns and encoding labels


def basicPreparation(fileName):
    file = pd.read_csv(fileName)
    label_encoder = preprocessing.LabelEncoder()
    file['label'] = label_encoder.fit_transform(
        file['label'])  # labelling data FAKE = 0 REAL = 1
    file = file.applymap(lambda s: s.lower() if type(
        s) == str else s)  # set every letter to lowercase
    return (file['text'], file['label'])


def dataPreprocessing(articles, labels):
    deletionIndexes = []
    for articleIndex in range(len(articles)):
        articles[articleIndex] = ''.join(w+' ' for w in articles[articleIndex].split(
            ' ') if not w in stop_words and w != '')  # remove all stopwords
        # remove all special characters
        articles[articleIndex] = re.sub(
            r'[^a-zA-Z]+', ' ', articles[articleIndex])
        articleLength = len(articles[articleIndex])
        if(articleLength == 0 or articleLength < 500 or articleLength > 5000):
            # save indexes of rows that need to be deleted
            deletionIndexes.append(articleIndex)
    articles = articles.drop(deletionIndexes, axis=0)
    labels = labels.drop(deletionIndexes, axis=0)
    return (articles, labels)


#Prepare data
(text, labels) = basicPreparation("news.csv")
(text, labels) = dataPreprocessing(text, labels)

X_train, X_test, y_train, y_test = train_test_split(
    text, labels, test_size=0.40, random_state=42)

for size in NgramSizeRange:
    extractedData["bagOfWords"]["Ngram"].append(
        {"vectorizer": CountVectorizer(analyzer="char", ngram_range=(size, size))})
    extractedData["tfidf"]["Ngram"].append(
        {"vectorizer": TfidfVectorizer(analyzer="char", ngram_range=(size, size))})

extractedData["bagOfWords"]["Word"].append({"vectorizer": CountVectorizer(
    analyzer="word", ngram_range=(1, 1))})
extractedData["tfidf"]["Word"].append({"vectorizer": TfidfVectorizer(
    analyzer="word", ngram_range=(1, 1))})

print('{:^20}  {:^20}  {:^20} {:^20} {:^20} {:^20} {:^20}'.format(
    "Vectorization", "Word type", "Size", "Classifier", "Score", "Learn time", "Test time"))

for method in extractedData.keys():
    for wordType in extractedData[method].keys():
        for size in range(len(extractedData[method][wordType])):
            extractedData[method][wordType][size]["classificator"] = {
                "KNN": KNeighborsClassifier(),
                "RandomForest": RandomForestClassifier(),
                "Naive Bayes": MultinomialNB(),
                "SVC": svm.SVC(),
                "MLP": MLPClassifier()
            }

            extractedData[method][wordType][size]["vectorizer"].fit(X_train)

            trainingFeatures = extractedData[method][wordType][size]["vectorizer"].transform(
                X_train)
            testFeatures = extractedData[method][wordType][size]["vectorizer"].transform(
                X_test)

            scaler = StandardScaler(with_mean=False)
            trainingFeatures = scaler.fit_transform(trainingFeatures)
            testFeatures = scaler.fit_transform(testFeatures)

            for classifier in extractedData[method][wordType][size]["classificator"].keys():
                startLearn = time.time()
                extractedData[method][wordType][size]["classificator"][classifier].fit(
                    trainingFeatures, y_train)
                endLearn = time.time()

                startTest = time.time()
                acuraccy = extractedData[method][wordType][size]["classificator"][classifier].score(
                    testFeatures, y_test)*100
                endTest = time.time()

                with open('./Wyniki/' + classifier + '.csv', mode='a', newline='') as score_file:
                    csv_writer = csv.writer(score_file)
                    csv_writer.writerow([method, wordType, minNgram+size, classifier,
                                         str(round(acuraccy, 2)) + "%",
                                         str(round(endLearn-startLearn, 3)),
                                         str(round(endTest-startTest, 3))])

                print('{:^20}  {:^20}  {:^20} {:^20} {:^20} {:^20} {:^20}'.format(method, wordType, minNgram+size, classifier,
                                                                                  str(round(
                                                                                      acuraccy, 2)) + "%",
                                                                                  str(round(
                                                                                      endLearn-startLearn, 3)) + "s",
                                                                                  str(round(endTest-startTest, 3)) + "s"))

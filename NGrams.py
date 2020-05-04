import pandas as pd
from enum import Enum
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import copy
import time


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


def ngramVectorizationPreparation(articles):
    return

def wordVectorizationPreparation(articles):
    return


#Prepare data
(text, labels) = basicPreparation("news.csv") 

ngramText = ngramVectorizationPreparation(text)

wordsText = wordVectorizationPreparation(text)


# Extract information from data

#Vectorize data


start = time.time()

for size in range(1, 4):
    extractedData["bagOfWords"]["char"].append({"vectorizer": CountVectorizer(analyzer="char", ngram_range=(size, size))})
    extractedData["tfidf"]["char"].append({"vectorizer": TfidfVectorizer(analyzer="char", ngram_range=(size, size))})

extractedData["bagOfWords"]["word"].append({"vectorizer": CountVectorizer(
    analyzer="word", ngram_range=(1, 1))})
extractedData["tfidf"]["word"].append({"vectorizer": TfidfVectorizer(
    analyzer="word", ngram_range=(1, 1))})

for method in extractedData.keys():
    for wordType in extractedData[method].keys():
        for size in range(len(extractedData[method][wordType])):
            extractedData[method][wordType][size]["classificator"] = {
                "SVC": svm.SVC(),
                "KNN": KNeighborsClassifier(n_neighbors=5),
                "RandomForest": RandomForestClassifier()
            }
            for classifier in extractedData[method][wordType][size]["classificator"].keys():
                extractedData[method][wordType][size]["features"] = extractedData[method][wordType][size]["vectorizer"].fit_transform(text)
                X_train, X_test, y_train, y_test = train_test_split(extractedData[method][wordType][size]["features"], labels, test_size=0.60, random_state=42)
                extractedData[method][wordType][size]["classificator"][classifier].fit(X_train, y_train)
                print(method, wordType, size+1, classifier, round(extractedData[method][wordType][size]["classificator"][classifier].score(X_test, y_test), 2), "% correct")

end = time.time() / 60

print(end - start)
# print(extractedData)

# print(len(tfidfVectorizer.get_feature_names()))

#Train data

#Save trained models

#Plot data

x = [xaxis*xaxis for xaxis in range(200)]
y = [xaxis for xaxis in range(200)]

plt.plot(y, x)

plt.xlabel('x')
plt.ylabel('y')

plt.title('My first graph!')
plt.ion()
plt.savefig('./plots/file.png')

import pandas as pd
from enum import Enum
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
import copy


datasplit = {
    "ngram": [],
    "word": []
}

extractedData = {
    "bagOfWords": copy.deepcopy(datasplit),
    "tfidf": copy.deepcopy(datasplit),
    # "hashing": copy.deepcopy(datasplit)
}

info = {
    "vectorizer": {},
    "features": {},
    "classificator": {}
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




for size in range(1, 2):
    extractedData["bagOfWords"]["ngram"].append({"vectorizer": CountVectorizer(analyzer="char", ngram_range=(size, size))})
    extractedData["bagOfWords"]["word"].append({"vectorizer": CountVectorizer(analyzer="word", ngram_range=(size, size))})
    extractedData["tfidf"]["ngram"].append({"vectorizer": TfidfVectorizer(analyzer="char", ngram_range=(size, size))})
    extractedData["tfidf"]["word"].append({"vectorizer": TfidfVectorizer(analyzer="word", ngram_range=(size, size))})

for method in extractedData.keys():
    for wordType in extractedData[method].keys():
        for size in range(len(extractedData[method][wordType])):
            extractedData[method][wordType][size]["classificator"] = {}
            extractedData[method][wordType][size]["features"] = extractedData[method][wordType][size]["vectorizer"].fit_transform(text)
            X_train, X_test, y_train, y_test = train_test_split(extractedData[method][wordType][size]["features"], labels, test_size=0.60, random_state=42)
            extractedData[method][wordType][size]["classificator"]["SVC"] = svm.SVC()
            extractedData[method][wordType][size]["classificator"]["SVC"].fit(X_train, y_train)
            print(method, wordType, size, extractedData[method][wordType][0]["classificator"]["SVC"].score(X_test, y_test), "% correct")
# print(extractedData)
# extractedData["bagOfWords"]["word"][0]["features"] = extractedData["bagOfWords"]["word"][0]["vectorizer"].fit_transform(text)

# print(len(tfidfVectorizer.get_feature_names()))

#Train data
# X_train, X_test, y_train, y_test = train_test_split(extractedData["bagOfWords"]["word"][0]["features"], labels, test_size=0.50, random_state=42)


# clf = svm.SVC()
# clf.fit(X_train, y_train)
# print(clf.score(X_test, y_test))


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

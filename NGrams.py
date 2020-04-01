import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt


#Prepares data by removing not needed columns and encoding labels
def basicPreparation(fileName): 
    file = pd.read_csv(fileName)
    label_encoder = preprocessing.LabelEncoder()
    file['label'] = label_encoder.fit_transform(file['label']) #labelling data FAKE = 0 REAL = 1
    file = file.applymap(lambda s: s.lower() if type(s) == str else s) #set everything to lowercase
    return (file['text'], file['label'])

def wordVectorizationPreparation():
    print("h")

def ngramVectorizationPreparation():
    print("h")

def NgramVectorization():
    print("s")
def WordVectorization():
    print("s")


#Prepare data
(text, labels) = basicPreparation("news.csv") 


#Vectorize data
bagOfWordsVectorizer = CountVectorizer(analyzer="word", max_features=30)
tfidfVectorizer = TfidfVectorizer()

features = tfidfVectorizer.fit_transform(text)

print(tfidfVectorizer.get_feature_names())

#Train data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.50, random_state=42)

clf = svm.SVC()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))


#Save trained models


#Plot data

x = [1, 2, 3]
y = [2, 4, 1]

plt.plot(x, y)

plt.xlabel('x - axis')
plt.ylabel('y - axis')

plt.title('My first graph!')
plt.show()

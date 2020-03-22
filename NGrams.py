import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm


#Prepares data by removing not needed columns and encoding labels
def prepareDataset(fileName): 
    file = pd.read_csv(fileName)
    cols = [c for c in file.columns if c.lower() == 'text' or c.lower() == 'label']
    file = file[cols]
    label_encoder = preprocessing.LabelEncoder()
    file['label'] = label_encoder.fit_transform(file['label']) #labelling data FAKE = 0 REAL = 1
    file = file.applymap(lambda s: s.lower() if type(s) == str else s) #set everything to lowercase
    #TODO delete rows with small number of characters
    return file

def NgramVectorization():
    print("s")
def WordVectorization():
    print("s")

learningData = prepareDataset("news.csv") 

print(learningData)

# bagOfWordsVectorizer = TfidfVectorizer(max_features=1000)
# bagOfWordsData = learningData

# print(bagOfWordsData["text"])

# bagOfWordsData["text"] = bagOfWordsVectorizer.fit_transform(learningData["text"]).toarray()


# x_train, x_test, y_train, y_test = train_test_split(bagOfWordsData["text"], bagOfWordsData["label"], test_size=0.2)

# clf = svm.SVC()
# clf.fit(x_train, y_train)
# print(clf.score(y_train, y_test))

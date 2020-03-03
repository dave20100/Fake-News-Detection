import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


data = pd.read_csv("news.csv")


def createNGramArray(text, ngramLength = 3):
    return "string"

vectorizer = CountVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(data["text"])
analyze = vectorizer.build_analyzer()
print(analyze('bi-grams lol'))
print(X.toarray())
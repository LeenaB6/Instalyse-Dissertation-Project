import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify
import joblib


stopwords = stopwords.words('english') #using the pre-defined nltk stopwords for pre-processing of text to improve accuracy


def removeStopwords(data):
    newTextArr = []
    for word in data.split():
        if word.lower() not in stopwords:
            newTextArr.append(word.lower())
    newText = " ".join(newTextArr)

    return(newText)


def removePunc(data):
    punctuation = '''!()-+=£`|[]{};:'"\,<>./?@#$%^&*_~'''
    noPunc = ""
    for char in data:
        if char not in punctuation:
            noPunc = noPunc + char
    return(noPunc)
    

def removeURLs(data):
    noURL = re.sub('((www.[^s]+)|(https?://[^s]+))',' ',data) #Substituting urls for a blank space
    return(noURL)


def removeNums(data):
    noNum = re.sub('[0-9]+','',data)
    return noNum


tokenizer = RegexpTokenizer(r'\w+')

lemmatizer = nltk.WordNetLemmatizer()
def lemmatizeText(data):
    text = [lemmatizer.lemmatize(word) for word in data]
    return ' ' .join(text)


def trainTopicAI():
    dsColumns = ['text', 'topic']
    dsEncoding = "ISO-8859-1"
    topicDS = pd.read_csv("/Users/leenabhela/Documents/Documents – Leena’s MacBook Pro (5877)/University/Year 3/COMP390/Topic_Identification.csv", \
                            encoding = dsEncoding, names = dsColumns)
    topicDS.tail(5)
    topicDS.info()

    print(topicDS.head(5))
    dataset = topicDS.sample(frac = 1) #Shuffles all of the dataset so topic data is mixed
    print(dataset.head(5))

    dataset['text'] = dataset['text'].apply(removeStopwords)
    print(dataset.head(5))

    dataset['text'] = dataset['text'].apply(removePunc)
    print(dataset.head(5))

    dataset['text'] = dataset['text'].apply(removeURLs)
    print(dataset.head(5))

    dataset['text'] = dataset['text'].apply(removeNums)
    print(dataset.head(5))

    dataset['text'] = dataset['text'].apply(tokenizer.tokenize)

    dataset['text'] = dataset['text'].apply(lambda text: lemmatizeText(text))
    print(dataset.head(5))

    x_train, x_test, y_train, y_test = train_test_split(dataset['text'], dataset['topic'], test_size=0.3)
    count_vect = CountVectorizer()
    x_train_counts = count_vect.fit_transform(x_train)
    x_test_counts = count_vect.transform(x_test)
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
    x_test_tfidf = tfidf_transformer.transform(x_test_counts)

    nb = MultinomialNB()
    nb.fit(x_train_tfidf, y_train)
    y_nb_pred = nb.predict(x_test_tfidf)
    acc_score = accuracy_score(y_nb_pred, y_test)
    print("accuracy of topic identification ai model = ", acc_score)

    fileName = "topicAI.joblib"
    joblib.dump(nb, fileName)

    vecFile = "vecTopicAI.joblib"
    joblib.dump(count_vect, vecFile)


nb = joblib.load("topicAI.joblib")
vectorizer = joblib.load("vecTopicAI.joblib")

testText = "today i went to the forest, sometimes its just nice to just hear the leaves. so peaceful. #beach #sea"

#If want to re-train the AI on a different shuffle of data then uncomment out the following line:
# trainTopicAI()

def preProcessText(testText):
    testText = removeStopwords(testText)
    testText = removePunc(testText)
    testText = removeURLs(testText)
    testText = removeNums(testText)
    tokenizer = RegexpTokenizer(r'\w+')
    testText = tokenizer.tokenize(testText)
    testText = lemmatizeText(testText)
    return testText
testText = preProcessText(testText)

def getTopic(testText):    
    pred = nb.predict(vectorizer.transform([testText]))
    print("topic = ", pred)
    return pred
getTopic(testText)

app = Flask(__name__)

@app.route("/topic")
def flaskGetTopic():
    analysisText = request.args.get('caption')

    analysisText = preProcessText(analysisText)
    pred = getTopic(analysisText)
    pred = str(pred[0])
    print ("topic = ", pred)

    return jsonify(pred = str(pred))

if __name__ == '__main__':
    app.run(port=5001)
from sklearn import set_config
import pandas as pd
import numpy as np
import re
import nltk
import nltk.classify.util
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, request, jsonify
import joblib


stopwordList = stopwords.words('english') #using the pre-defined nltk stopwords for pre-processing of text to improve accuracy

print(stopwordList)
def removeStopwords(data): #removing the nltk stopwords from the dataset 

    newTextArr = []
    for word in data.split():
        if word.lower() not in stopwordList:
            newTextArr.append(word.lower())
    newText = " ".join(newTextArr)

    return(newText)

def removePunc(data): #removes punctuation from the dataset
    punctuation = '''!()-+=£`|[]{};:'"\,<>./?@#$%^&*_~'''
    noPunc = ""
    for char in data:
        if char not in punctuation:
            noPunc = noPunc + char
    return(noPunc)

def removeURLs(data): #removes urls from dataset
    noURL = re.sub('((www.[^s]+)|(https?://[^s]+))',' ',data) #Substituting urls for a blank space
    return(noURL)

def removeNums(data): #removes numbers from the dataset
    noNum = re.sub('[0-9]+','',data)
    return noNum



tokenizer = RegexpTokenizer(r'\w+') #splits a string sentence into separate words to be assigned sentiment value


lemmatizer = nltk.WordNetLemmatizer() #removes suffixes and prefixes from words in the dataset so they are left to be their 'root' word
def lemmatizeText(data):
    text = [lemmatizer.lemmatize(word) for word in data]
    return ' ' .join(text)


def trainSentiAI():
    twitterDSColumns = [ 'target', 'ids', 'date', 'query', 'user', 'text']
    twitterDSEncoding = "ISO-8859-1"
    twitterDS = pd.read_csv("/Users/leenabhela/Documents/Documents – Leena’s MacBook Pro (5877)/University/Year 3/COMP390/training.1600000.processed.noemoticon.csv", \
                            encoding = twitterDSEncoding, names = twitterDSColumns) #Reads in the data from the csv file
    twitterDS.tail(5)
    twitterDS.info()

    twitterDS['target'] = twitterDS['target'].replace(4,1) #Replacing the 4 that indicates positive sentiment with 1 so it's easier to distinguish between the two

    senti = {0:"Negative", 1:"Positive"} #Declaring what is percieved as positive and negative sentiment based on the 'target' data

    dataNeeded = twitterDS[['text', 'target']] #Only fields needed in dataset to train my ai model
    print(dataNeeded.head(2))

    posData = dataNeeded[dataNeeded['target'] == 1] #Changes to a binary target data system, 1 fro positive and 0 for negative
    negData = dataNeeded[dataNeeded['target'] == 0]


    posData = posData.iloc[0:1001] #Gets values in data up to index 1000
    negData = negData.iloc[0:1001]

    dataset = pd.concat([posData, negData]) #Creating a dataset to train and test AI
    dataset = dataset.sample(frac = 1) #Shuffles all of the dataset so positve and negative data is mixed
    print(dataset.head(5))

    dataset['text'] = dataset['text'].apply(removeStopwords) #textual preprocessing called on the data
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


    x_train, x_test, y_train, y_test = train_test_split(dataset['text'], dataset['target'], stratify = dataset['target'], test_size=0.3, random_state=42)

    vec = CountVectorizer(stop_words='english')
    x = vec.fit_transform(x_train).toarray() #transforming the training data to be a Count matrix
    x_test = vec.transform(x_test).toarray() #transforming the test data to be a Count matrix

    nb = MultinomialNB() #declares the algorithm being used
    nb.fit(x, y_train) #fits the model with the x and y training data (text and target data)

    probs = nb.predict_proba(x_test) 

    acc = nb.score(x_test, y_test) #accuracy score of the ai model
    print("accuracy of sentiment ai model ", acc)

    fileName = "simpleAI.joblib" #saves model to a file
    joblib.dump(nb, fileName)

    vecFile = "vecSimpleAI.joblib"
    joblib.dump(vec, vecFile)


nb = joblib.load("simpleAI.joblib") #loads model from file
vectorizer = joblib.load("vecSimpleAI.joblib")

testText = "i love you, you are amazing and my best friend!"

#If want to re-train the AI on a different shuffle of data then uncomment out the following line:
#trainSentiAI()

def preProcessText(testText): #conducts all of the pre-processing functions on the text passed
    testText = removeStopwords(testText)
    testText = removePunc(testText)
    testText = removeURLs(testText)
    testText = removeNums(testText)
    tokenizer = RegexpTokenizer(r'\w+')
    testText = tokenizer.tokenize(testText)
    testText = lemmatizeText(testText)
    return testText #returns the updated text
testText = preProcessText(testText)

def analyseText(testText): #analyse the inputted text
    vectorizedText = vectorizer.transform([testText]) #transforms the text using the vectorizer
    pred = nb.predict(vectorizedText) #predicts the sentiment of the inputted text (1 or 0)
    print("predicted sentiment of given text = ", pred)
    percent = nb.predict_proba(vectorizedText) #gets the percentages of how positive or negative it is
    percent = np.array(percent) #converts it to a numpy array
    print("predicted percentage [negative, positive] = ", percent)
    if (percent[0][0] > percent[0][1]): #sees if the negative percentage is bigger than the positive one
        percent = percent[0][0] #saves the negative percentage if it is bigger
    else:
        percent = percent[0][1] #saves the positive percentage if its bigger
    print("predicted percentage = ", percent)

    return pred[0], percent #returns the predicted sentiment and the percentage of how positive or negative it is
analyseText(testText)

app = Flask(__name__)

@app.route("/senti") #runs the following function when called via a flask application
def flaskGetAnalysis():
    analysisText = request.args.get('caption')

    analysisText = preProcessText(analysisText)

    pred, percent = analyseText(analysisText)
    print("pred = ", pred)
    print("perc = ", percent)

    return jsonify(pred = int(pred), percent = float(percent))

if __name__ == '__main__': #runs on port 5000
    app.run(port=5000) 
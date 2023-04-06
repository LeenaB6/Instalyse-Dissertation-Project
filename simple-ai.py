import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
import collections
#nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
from collections import OrderedDict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import punkt
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
# import keras as keras

twitterDSColumns = [ 'target', 'ids', 'date', 'query', 'user', 'text']
twitterDSEncoding = "ISO-8859-1"
twitterDS = pd.read_csv("/Users/leenabhela/Documents/Documents – Leena’s MacBook Pro (5877)/University/Year 3/COMP390/training.1600000.processed.noemoticon.csv", \
                        encoding = twitterDSEncoding, names = twitterDSColumns)
twitterDS.tail(5)
twitterDS.info()

twitterDS['target'] = twitterDS['target'].replace(4,1) #Replacing the 4 that indicates positive sentiment with 1 so it's easier to distinguish between the two

senti = {0:"Negative", 1:"Positive"} #Declaring what is percieved as positive and negative sentiment based on the 'target' data

dataNeeded = twitterDS[['text', 'target']] #Only fields needed in dataset to 
print(dataNeeded.head(2))

posData = dataNeeded[dataNeeded['target'] == 1] #Changes to a binary target data system
negData = dataNeeded[dataNeeded['target'] == 0]

# pos_data = pos_data.iloc[0:500001] #Gets values in data up to index 500000
# neg_data = neg_data.iloc[0:500001]

posData = posData.iloc[0:101] #Gets values in data up to index 100
negData = negData.iloc[0:101]

dataset = pd.concat([posData, negData]) #Creating a dataset to train and test AI
dataset = dataset.sample(frac = 1) #Shuffles all of the dataset so positve and negative data is mixed
print(dataset.head(5))

stopwords = stopwords.words('english') #using the pre-defined nltk stopwords for pre-processing of text to improve accuracy

print(stopwords)

def removeStopwords(data):
    #text = "When I first met her she was very quiet. She remained quiet during the entire two hour long journey from Stony Brook to New York."

    newTextArr = []
    for word in data.split():
        if word.lower() not in stopwords:
            newTextArr.append(word.lower())
    newText = " ".join(newTextArr)

    return(newText)

dataset['text'] = dataset['text'].apply(removeStopwords)

print(dataset.head(5))

def removePunc(data):
    punctuation = '''!()-+=£`|[]{};:'"\,<>./?@#$%^&*_~'''
    noPunc = ""
    for char in data:
        if char not in punctuation:
            noPunc = noPunc + char
    return(noPunc)
    # data['text'] = data['text'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '' , x))
    
dataset['text'] = dataset['text'].apply(removePunc)
print(dataset.head(5))

def removeURLs(data):
    noURL = re.sub('((www.[^s]+)|(https?://[^s]+))',' ',data) #Substituting urls for a blank space
    return(noURL)

dataset['text'] = dataset['text'].apply(removeURLs)
print(dataset.head(5))

def removeNums(data):
    noNum = re.sub('[0-9]+','',data)
    return noNum

dataset['text'] = dataset['text'].apply(removeNums)
print(dataset.head(5))

# def removeHandles(data):

# def removePatterns(data, pattern):
#     patternArr = re.findall(pattern, input_txt)
#     for word in patternArr:
#         input_txt = re.sub(word, "", input_txt)
#     return input_txt

tokenizer = RegexpTokenizer(r'\w+')
dataset['text'] = dataset['text'].apply(tokenizer.tokenize)

stemmer = nltk.PorterStemmer() #why lamda
def StemText(data):
    text = [stemmer.stem(word) for word in data]
    return text
dataset['text']= dataset['text'].apply(lambda x: StemText(x))
print(dataset.head(5))

lemmatizer = nltk.WordNetLemmatizer()
def lemmatizeText(data):
    text = [lemmatizer.lemmatize(word) for word in data]
    return ' ' .join(text)
dataset['text'] = dataset['text'].apply(lambda text: lemmatizeText(text))
print(dataset.head(5))

df = dataset
x_train, x_test, y_train, y_test = train_test_split(df['text'], df['target'], random_state=0)


# x_train, x_test, y_train, y_test = train_test_split(dataset['text'], dataset['target'], test_size = 0.3)
# df = dataset

# tfidf = TfidfVectorizer(max_df=0.90, min_df=0.02, max_features=1000, stop_words='english') #GO BACK TO THIS AND LAMBDAS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# # vectorizer = TfidfVectorizer()
# tfidf.fit(list(x_train) + list(x_test))
# x_train_tfidf = tfidf.transform(x_train)
# x_test_tfidf = tfidf.transform(x_test)

# nb = MultinomialNB()
# nb.fit(x_train_tfidf, y_train)
# y_pred_nb = nb.predict(x_test_tfidf)
# print('naive bayes tfidf accuracy %s' % accuracy_score(y_pred_nb, y_test))


fileName = "simpleAI.joblib"
joblib.dump(nb, fileName)

vecFile = "vecSimpleAI.joblib"
joblib.dump(tfidf, vecFile)


nb = joblib.load("simpleAI.joblib")
vectorizer = joblib.load("vecSimpleAI.joblib")

testText = "i hate you, you are stupid and my ultimate enemy!"
def preProcessText(testText):
    testText = removeStopwords(testText)
    testText = removePunc(testText)
    testText = removeURLs(testText)
    testText = removeNums(testText)
    tokenizer = RegexpTokenizer(r'\w+')
    testText = tokenizer.tokenize(testText)
    testText = StemText(testText)
    testText = lemmatizeText(testText)
    return testText
testText = preProcessText(testText)

vectorizedText = vectorizer.transform([testText])
pred = nb.predict(vectorizedText)
print("pred senti = ", pred)

#Code onwards is for Topic Identification

# def transformData(data, fieldName, tokenizer, weight):
#     topicVectorizer = TfidfVectorizer(tokenizer=tokenizer)
#     tVRes = topicVectorizer.fit_transform(data[fieldName])
#     return(topicVectorizer, tVRes)

# topicVectorizer, tVRes = transformData(dataset, fieldName="text", tokenizer = word_tokenize, weight="TFIDF")
# featureNames = topicVectorizer.get_feature_names_out()

# text_collection = OrderedDict([(index, text) for index, text in enumerate(dataset["text"])])
# corpus_index = [n for n in text_collection]
# topicDS = pd.DataFrame(tVRes.todense(), index=corpus_index, columns=featureNames)


# def LDATopicModelling(topicDS, topicNum, topWords):
#     featureNames = list(topicDS)
#     model = LatentDirichletAllocation(n_components=topicNum, max_iter = 10,
#                                       learning_method = 'online',
#                                       learning_decay = 0.9,
#                                       random_state = 4)
#     model.fit(topicDS)

#     for index, topic in enumerate(model.components_):
#         message = "\nTopic #{}:".format(index)
#         message += " ".join([featureNames[i] for i in topic.argsort()[:-topWords - 1 :-1]])
#         print(message)
#         print("="*70)

# LDATopicModelling(topicDS, 5, 2)
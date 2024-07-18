import math
from flask import Flask, request
import pandas as pd
import numpy as np
import heapq
from sentence_transformers import SentenceTransformer
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, support_credentials=True)

def setup_app(app):
    global model, sentences, df
    model = SentenceTransformer('firqaaa/indo-sentence-bert-base')
    df = pd.read_csv('./data/intents-2.csv', names=["pattern", 'tag', 'response'], dtype=str)
    sentences = []
    for i in range(len(df["pattern"])):
        sentences.append(df["pattern"].values[i])
setup_app(app)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/chatbot/single-context", methods=['POST'])
def chat():
    sentence = model.encode(request.form['sentence'])
    embeddings = model.encode(sentences)
    similarityResult = np.array(model.similarity(sentence, embeddings))
    highestIndex = np.argmax(similarityResult)
    highestVal = np.max(similarityResult)
    if(highestVal >= 0.5):
        print("\n[MODEL DEBUG LOG]")
        print("Similarity result: ", similarityResult)
        print("Highest index: ", highestIndex)
        print("Highest value: ", highestVal)
        print("Most similar sentence: ", sentences[highestIndex])
        print("Response: ", df["response"][highestIndex], "\n")
    else:
        return  "Gatau ah cape"
    return  df["response"][highestIndex]

@app.route("/chatbot/multi-context", methods=['POST'])
def multiContext():
    sentence = model.encode(request.form['sentence'])
    embeddings = model.encode(sentences)

    result = model.similarity(sentence, embeddings)
    similarityResult = np.array(result)
    
    highestIndex = np.argmax(similarityResult)
    highestVal = np.max(similarityResult)
    
    mostSimilars = -np.sort(-result)
    # mostSimilarsArr = mostSimilars.reshape(-1)
    
    # Get 3 highest value's indexes
    indices = np.argpartition(similarityResult, -3)[-3:]
    print(indices)
    tags = []
    for data in indices:
        tags.append(df["tag"][data])
    
    if(highestVal >= 0.5):
        print("\n[MODEL DEBUG LOG]")
        print("Result Similarity:\n", mostSimilars)
        print("\nTop 3 Highest Index: ", indices)
        # print("Similarity result: ", similarityResult)
        print("Highest index: ", highestIndex)
        # print("Highest value: ", highestVal)
        # print("Most similar sentence: ", sentences[highestIndex])
        # print("Response: ", df["response"][highestIndex])
        # print("3 highest: ", heapq.nlargest(3, range(len(similarityResult)), similarityResult.__getitem__))
        print("Tag: ", df["tag"][highestIndex], "\n")
    else:
        return  "Gatau ah cape"
    return  df["response"][highestIndex]

if __name__ == '__main__':
    app.run(host='192.168.50.106', port=5555, debug=True)
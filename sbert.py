# Importing Packages
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# os.system('cls')

# Load model
model = SentenceTransformer('firqaaa/indo-sentence-bert-base')

# Setting up dataframe for sentences
df = pd.read_csv('./data/intents.csv', names=["pattern", 'tag', 'response'], dtype=str)
sentences = []
for i in range(len(df["pattern"])):
    sentences.append(df["pattern"].values[i])
# print(sentences)

# Encoding sentences and sentence (sentence is designated for input from user)
sentence = model.encode("Halo! Selamat pagi!")
embeddings = model.encode(sentences)
# print(embeddings)

# Checking similarity
similarityResult = np.array(model.similarity(sentence, embeddings))

# Getting highest value and its index
highestIndex = np.argmax(similarityResult)
highestVal = np.max(similarityResult)

# Handling if the inputs' similarity score is higher than 50%
if(highestVal >= 0.5):
    print("Similarity result: ", similarityResult)
    print("Highest index: ", highestIndex)
    print("Highest value: ", highestVal)
    print("Most similar sentence: ", sentences[highestIndex])
    print("Response: ", df["response"][highestIndex])

# Handling if the inputs' similarity score is lower than 50%
else:
    print("We can't provide any answer for your question right noe since this is a new knowledge for us.")
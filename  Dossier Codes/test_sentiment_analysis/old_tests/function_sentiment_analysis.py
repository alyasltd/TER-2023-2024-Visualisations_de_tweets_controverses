import pandas as pd
from pickle import FALSE
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

"""
Ce fichier contient une fonction pour prétraiter le texte attribut dans les fichiers JSON. 
La fonction analyze_sentiment_and_save crée un DataFrame de tweets avec une analyse de sentiment
à partir du modèle BERT.
La fonction prends en parametres un fichier json, le nom d'un fichier json et permet de créer un 
fichier csv avec les résultat sur un tweet. 
C'était le début de mon analyse.  
"""

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def analyze_sentiment_and_save(json_data, filename, dataframe=None):
    # Load the model, tokenizer, and configuration
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    # Initialize DataFrame if it doesn't exist
    columns = ["id", "positive", "neutral", "negative"]
    try:
        dataframe = pd.read_csv('sentiment_results.csv')
    except FileNotFoundError:
        dataframe = pd.DataFrame(columns=columns)

    # Check if "text" key exists in json_data
    if "text" in json_data:
        # Apply preprocessing to the text
        text = preprocess(json_data["text"])

        # Tokenize and classify
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output.logits.detach().numpy()
        scores = softmax(scores)

        # Create a dictionary with the results
        result = {"id": filename}
        for i in range(scores.shape[1]):
            l = config.id2label[i]
            s = scores[0][i]
            result[l] = np.round(float(s), 4)

        # Append the results to the DataFrame
        dataframe = pd.concat([dataframe, pd.DataFrame([result])], ignore_index=True)

    return dataframe


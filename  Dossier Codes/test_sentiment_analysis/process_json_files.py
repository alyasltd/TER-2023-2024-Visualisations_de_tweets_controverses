import os
import pandas as pd
from multiprocessing import Pool
import numpy as np
import timeit
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from scipy.special import softmax

"""
Ce fichier contient une fonction pour prétraiter le texte attribut dans les fichiers JSON. 
La fonction process_json_file prends en paramètres le path d'un fichier json, 
et appliquer la fonction analyse_sentiment_batch sur celui ci, cette fonction process UN json file que elle 
ajoute à une variable json data. 
Le principe de la fonction batch est que elle prends en entrée un batch de fichier json, 
elle charge le model bert d'analyse des sentiments, puis elle applique le model sur ce batch. 
Ce batch est ensuite ajouter à la variable result. 
Et enfin le script python  A FINIR 
"""

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Update the path to your folder containing JSON files
json_folder = r"/home/ter_meduse/our_data/tweets_chloroquine"

def process_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    # Check if the "text" attribute is present
    if "text" not in json_data:
        print(f"Skipping file {file_path} as it does not contain a 'text' attribute.")
        return None

    # Apply sentiment analysis function
    results = analyze_sentiment_batch([json_data])

    return results  # Do not extract the result from the list

def analyze_sentiment_batch(batch):
    # Load the model, tokenizer, and configuration
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    # Apply preprocessing to the text in the batch
    texts = [preprocess(example["text"]) for example in batch]

    # Tokenize and classify the batch
    encoded_inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    output = model(**encoded_inputs)
    scores = output.logits.detach().numpy()
    scores = softmax(scores, axis=1)

    # Create a list of dictionaries with the results
    results = []
    for i, example in enumerate(batch):
        result = {
            "id": example["id"],
            "Text": preprocess(example["text"]),
            "positive": np.round(float(scores[i][config.label2id['positive']]), 4),
            "neutral": np.round(float(scores[i][config.label2id['neutral']]), 4),
            "negative": np.round(float(scores[i][config.label2id['negative']]), 4)
        }
        results.append(result)

    return results

#SCRIPT
if __name__ == "__main__":
    start_time = timeit.default_timer()

    # Get a list of all JSON files in the specified folder
    json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]

    # Create a pool of processes for parallel processing
    with Pool() as pool:
        # Process each JSON file in parallel
        results_list = pool.map(process_json_file, [os.path.join(json_folder, file) for file in json_files])

    # Remove None values (files without 'text' attribute)
    results_list = [result for result in results_list if result is not None]

    # Flatten the list of lists
    results_flat = [item for sublist in results_list for item in sublist]

    # Combine dictionaries into a single DataFrame
    results_df = pd.DataFrame(results_flat)

    # Save the results to a CSV file
    results_csv_path = r"/home/ter_meduse/analyse_alya/tweets_chloroquine.csv"
    results_df.to_csv(results_csv_path, index=False)

    elapsed_time = timeit.default_timer() - start_time
    print(f"Sentiment analysis results saved to: {results_csv_path}")
    print(f"Elapsed time: {elapsed_time} seconds")

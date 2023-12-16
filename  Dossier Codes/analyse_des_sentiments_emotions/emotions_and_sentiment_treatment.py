import os
import pandas as pd
from multiprocessing import Pool
import numpy as np
import timeit
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from scipy.special import softmax
from transformers import pipeline

"""
tweets_cancer_cannabis  
tweets_cancer_fasting   
tweets_chloroquine
done tweets_cancer_sport
done tweets_lithotherapy
"""

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Function to process a JSON file
def process_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    # Check if the "text" attribute is present
    if "text" not in json_data:
        print(f"Skipping file {file_path} as it does not contain a 'text' attribute.")
        return None

    # Apply sentiment analysis function
    sentiment_results = analyze_sentiment_batch([json_data])

    # Apply emotion analysis function
    emotion_results = analyze_emotion_batch([json_data])

    # Combine sentiment and emotion results
    results = {**sentiment_results[0], **emotion_results[0]}

    return results

# Function to analyze sentiment for a batch of examples
def analyze_sentiment_batch(batch):
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    texts = [preprocess(example["text"]) for example in batch]

    encoded_inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    output = model(**encoded_inputs)
    scores = output.logits.detach().numpy()
    scores = softmax(scores, axis=1)

    result = {
        "id": batch[0]["id"],
        "Text": preprocess(batch[0]["text"]),
        "positive": np.round(float(scores[0][config.label2id['positive']]), 4),
        "neutral": np.round(float(scores[0][config.label2id['neutral']]), 4),
        "negative": np.round(float(scores[0][config.label2id['negative']]), 4)
    }

    return [result]

# Function to analyze emotion for a batch of examples
def analyze_emotion_batch(batch):
    #local_emotion_model_dir = r"C:\Users\alyas\Desktop\TER\test_sentiment_analysis\model_emotions"
    local_emotion_model_dir = "/home/ter_meduse/analyse_alya/model_emotions"
    #model_name = "cardiffnlp/twitter-roberta-base-emotion-multilabel-latest"
    model = AutoModelForSequenceClassification.from_pretrained(local_emotion_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(local_emotion_model_dir)
    labels = model.config.id2label

    texts = [preprocess(example["text"]) for example in batch]

    inputs = tokenizer(texts, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    scores = logits.sigmoid().tolist()[0]

    result = {
        "id": batch[0]["id"],
        "Text": preprocess(batch[0]["text"]),
    }

    # Extract label and score from the top-k predictions
    for i, score in enumerate(scores):
        result[labels[i]] = score

    return [result]

# Main script
if __name__ == "__main__":
    start_time = timeit.default_timer()

    #json_folder = r"/home/ter_meduse/our_data/tweets_cancer_fasting"
    json_folder = r"/home/ter_meduse/analyse_alya/tweets_cancer_cannabis_alya"
    #json_folder = r"C:\Users\alyas\Desktop\TER\test_sentiment_analysis\tweets_lithotherapy"

    # Get a list of all JSON files in the specified folder
    json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]


    # Create a pool of processes for parallel processing
    with Pool() as pool:
        # Process each JSON file in parallel
        results_list = pool.map(process_json_file, [os.path.join(json_folder, file) for file in json_files])

    # Remove None values (files without 'text' attribute)
    results_list = [result for result in results_list if result is not None]

    # Combine dictionaries into a single DataFrame
    results_df = pd.DataFrame(results_list)

    # Save the results to a CSV file
    results_csv_path = r"/home/ter_meduse/analyse_alya/tweets_cancer_cannabis_all.csv"
    #results_csv_path = r"C:\Users\alyas\Desktop\TER\test_sentiment_analysis\tweets_litho_combined.csv"
    results_df.to_csv(results_csv_path, index=False)

    elapsed_time = timeit.default_timer() - start_time
    print(f"Combined analysis results saved to: {results_csv_path}")
    print(f"Elapsed time: {elapsed_time} seconds")

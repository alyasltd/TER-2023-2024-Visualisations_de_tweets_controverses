import os
import pandas as pd
from multiprocessing import Pool
import numpy as np
import timeit
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from scipy.special import softmax
import torch  # Import torch to manage GPU operations

# Check if CUDA (GPU support) is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Initialize models and tokenizers globally
MODEL_SENTIMENT = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer_sentiment = AutoTokenizer.from_pretrained(MODEL_SENTIMENT)
model_sentiment = AutoModelForSequenceClassification.from_pretrained(MODEL_SENTIMENT).to(device)

local_emotion_model_dir = r"/home/data/ter_meduse_log/analyse_alya/model_emotions"
tokenizer_emotion = AutoTokenizer.from_pretrained(local_emotion_model_dir)
model_emotion = AutoModelForSequenceClassification.from_pretrained(local_emotion_model_dir).to(device)

# Function to process a JSON file
def process_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    if "text" not in json_data:
        print(f"Skipping file {file_path} as it does not contain a 'text' attribute.")
        return None

    # Apply sentiment and emotion analysis functions
    sentiment_results = analyze_sentiment_batch([json_data])
    emotion_results = analyze_emotion_batch([json_data])
    results = {**sentiment_results[0], **emotion_results[0]}
    return results

# Updated functions for sentiment and emotion analysis
def analyze_sentiment_batch(batch):
    texts = [preprocess(example["text"]) for example in batch]
    encoded_inputs = tokenizer_sentiment(texts, return_tensors='pt', padding=True, truncation=True, max_length=1000).to(device)
    with torch.no_grad():
        output = model_sentiment(**encoded_inputs)
    scores = softmax(output.logits.cpu().detach().numpy(), axis=1)
    result = {"id": batch[0]["id"], "Text": preprocess(batch[0]["text"]),
              "positive": np.round(float(scores[0][2]), 4),  # Update indices based on your model's configuration
              "neutral": np.round(float(scores[0][1]), 4),
              "negative": np.round(float(scores[0][0]), 4)}
    return [result]

def analyze_emotion_batch(batch):
    texts = [preprocess(example["text"]) for example in batch]
    inputs = tokenizer_emotion(texts, return_tensors="pt",padding=True, truncation=True, max_length=1000).to(device)
    with torch.no_grad():
        outputs = model_emotion(**inputs)
    logits = outputs.logits.cpu()
    scores = logits.sigmoid().tolist()[0]
    result = {"id": batch[0]["id"], "Text": preprocess(batch[0]["text"])}
    for i, score in enumerate(scores):
        result[model_emotion.config.id2label[i]] = score
    return [result]

# Main script
if __name__ == "__main__":
    start_time = timeit.default_timer()
    json_folder = r"/home/data/ter_meduse_log/our_data/tweets_cancer_fasting"
    json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]
    with Pool() as pool:
        results_list = pool.map(process_json_file, [os.path.join(json_folder, file) for file in json_files])
    results_list = [result for result in results_list if result is not None]
    results_df = pd.DataFrame(results_list)
    results_csv_path = r"/home/data/ter_meduse_log/analyse_alya/fasting_all.csv"
    results_df.to_csv(results_csv_path, index=False)
    elapsed_time = timeit.default_timer() - start_time
    print(f"Combined analysis results saved to: {results_csv_path}")
    print(f"Elapsed time: {elapsed_time} seconds")

from function_sentiment_analysis import analyze_sentiment_and_save
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig  # Add this line
import numpy as np
from scipy.special import softmax
import os
import json

# Path to the folder containing JSON files
json_folder = r'C:\Users\alyas\Desktop\TER\test projet\test_run_sentiment_analysis'
output_folder = r'C:\Users\alyas\Desktop\TER\test projet'  # Adjust this to your desired output folder
# à changer 

# List all JSON files in the folder
json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
dataframe = pd.DataFrame()

# Iterate through each JSON file and apply the sentiment analysis function
for json_file in json_files:
    # Create the full path to the JSON file
    json_path = os.path.join(json_folder, json_file)

    # Read JSON data from the file
    with open(json_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

        # Apply the sentiment analysis function
        dataframe = analyze_sentiment_and_save(json_data, json_file, dataframe)

# Save the DataFrame to a CSV file
dataframe.to_csv('sentiment_results.csv', index=False)

# Move the output CSV file to the desired output folder
output_csv_path = os.path.join(output_folder, 'sentiment_results.csv')
os.rename('sentiment_results.csv', output_csv_path)

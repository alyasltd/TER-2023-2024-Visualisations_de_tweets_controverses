from pickle import FALSE
import torch
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Your JSON data
json_data = {
    "id": "120328370310479873",
    "creation": 1317523493.0,
    "user_id": "230883057",
    "social_network": "twitter",
    "nsfw": False,
    "request": ["d47670bd-00d9-4b4b-b654-af12857ee504"],
    "metrics": {
        "d47670bd-00d9-4b4b-b654-af12857ee504": {
            "retweet_count": 0,
            "reply_count": 0,
            "like_count": 0,
            "quote_count": 0
        }
    },
    "text": "Cannabis Ingredient Helps Cancer Patients Regain Appetite and Taste http://t.co/IBpv7NKe via @herbalhealthsys #Cancer #MMJ #Cannabis #mmot",
    "first_save": 1635509975.555947,
    "hashtags": ["#CANCER", "#MMJ", "#CANNABIS", "#MMOT"]
}

# Load the model and tokenizer
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Preprocess text
text = preprocess(json_data["text"])

# Tokenize and classify
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
scores = output.logits.detach().numpy()
scores = softmax(scores)

# Print labels and scores
ranking = np.argsort(scores)
ranking = ranking[::-1]
for i in range(scores.shape[1]):
    l = config.id2label[i]
    s = scores[0][i]
    print(f"{l}: {np.round(float(s), 4)}")

# Extract sentiment scores as labels
labels = np.argmax(scores, axis=1)

# ... (your existing code)

# Tokenize and classify
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
scores = output.logits.detach().numpy()
scores = softmax(scores)

# Print labels and scores
ranking = np.argsort(scores)
ranking = ranking[::-1]
predicted_label = ranking[0]  # Assuming the highest-scored label is the predicted sentiment
print(f"Predicted sentiment: {config.id2label[predicted_label]}")

# Use t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(hidden_states[0].numpy())

# Visualize the t-SNE results
plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=predicted_label, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.show()

pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Extract sentiment scores as labels
labels = np.argmax(scores, axis=1)

# Get the hidden states from the model
with torch.no_grad():
    output = model(**encoded_input)
    hidden_states = output.last_hidden_state

# Use t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(hidden_states[0].numpy())

# Visualize the t-SNE results
plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=labels, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.show()
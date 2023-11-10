from pickle import FALSE

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax

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

 # extraire feature spaces
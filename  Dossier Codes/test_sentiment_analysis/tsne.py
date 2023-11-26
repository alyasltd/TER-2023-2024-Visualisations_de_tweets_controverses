import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# chargement du model et du tokenizer
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, output_hidden_states=True)

# chargement du csv
csv_file_path = r"C:\Users\alyas\Desktop\TER\test_sentiment_analysis\all.csv"
df = pd.read_csv(csv_file_path)

# Initialisation de listes pour stocker les hidden states & labels
hidden_states_list = []
labels = []

# eteration sur chaque tweet du df
for index, row in df.iterrows():
    # Tokenize and classifier
    encoded_input = tokenizer(row["Text"], return_tensors='pt')
    output = model(**encoded_input)
    
    # Extraction des hidden states pour le tweet
    hidden_states = output.hidden_states[-1][0, 0, :].detach().numpy()
    
    # ajout 
    hidden_states_list.append(hidden_states)
    labels.append(row["Sentiment"])

# Convertion de la liste en liste numpy
hidden_states_array = np.array(hidden_states_list)


# Perform t-SNE on the hidden states
tsne = TSNE(n_components=2, perplexity= 2300, random_state=42)
# Perform t-SNE on the hidden states
print("Number of samples:", hidden_states_array.shape[0])
print("Perplexity value:", tsne.perplexity)
embedded_states = tsne.fit_transform(hidden_states_array)
embedded_states = tsne.fit_transform(hidden_states_array)

# Plot les embedded states avec differentes couleurs pour chaque sentiment
plt.figure(figsize=(10, 8))
for sentiment in df["Sentiment"].unique():
    indices = df[df["Sentiment"] == sentiment].index
    plt.scatter(embedded_states[indices, 0], embedded_states[indices, 1], label=sentiment)

plt.title('t-SNE Visualization of Hidden States')
plt.legend()

# Sauvegarde de la figure comme une image 
plt.savefig('tsne_plot.png')

# Affichage de la figure
plt.show()
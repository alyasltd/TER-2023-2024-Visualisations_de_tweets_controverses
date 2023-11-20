"""
Ce code est un script Python qui effectue plusieurs tâches liées à la lecture, l'analyse et la manipulation de données JSON, 
en particulier des tweets concernant les vaccins contre la COVID-19.En résumé, ce code analyse le contenu des tweets liés aux 
vaccins contre la COVID-19, en extrayant des informations textuelles et des métriques, puis les sauvegarde dans un fichier CSV pour une utilisation ultérieure.

Il peut etre adapter en modifiant le chemin_dossier, si on veut analyser des tweets sur la chloroquine on remplace par le chemin vers le dosqsier des tweets chloroquine.
"""

import json
import pandas as pd
import os
from textblob import TextBlob

# Fonction pour lire et parser le contenu d'un fichier JSON
def lire_et_parser_json(chemin):
    """
    Fonction permettant de lire le contenu d'un fichier JSON donné
    Elle gère les erreurs potentielles telles que les erreurs de décodage JSON et les cas où le fichier spécifié est introuvable.
    chemin : chaine de caractere correspondant au chemin du dossier a lire
    """
     
    try:
        with open(chemin, 'r', encoding='utf-8') as fichier:
            return json.load(fichier)
    except json.JSONDecodeError as e:
        print(f"Erreur de décodage JSON : {e}")
        return None
    except FileNotFoundError as e:
        print(f"Fichier non trouvé : {e}")
        return None

colonnes = ['Fichier', 'ID', 'Nb retweet', 'Nb like', 'Nb réponses', 'Nb citations', 'Hashtags', 'Texte', 'Mots', 'Phrases', 'Tags POS', 'Sentiment', 'Polarité', 'Subjectivité', 'Phrases nominales', 'Texte corrigé', 'Texte tokénizé']
df = pd.DataFrame(columns=colonnes)
data_list = []
chemin_dossier = "../../data/tweets_covid_vaccine"


for i in os.listdir(chemin_dossier):
    chemin_complet = os.path.join(chemin_dossier, i)
    data = lire_et_parser_json(chemin_complet)
    if data and 'text' in data and data['text'].strip():
        tt = TextBlob(data['text'])
            # Initialisation des variables de métriques
        retweet_count = reply_count = like_count = quote_count = 0

        if 'metrics' in data and data['metrics']:
            metric_key = list(data['metrics'].keys())[0]
            retweet_count = data['metrics'][metric_key].get('retweet_count', 0)
            reply_count = data['metrics'][metric_key].get('reply_count', 0)
            like_count = data['metrics'][metric_key].get('like_count', 0)
            quote_count = data['metrics'][metric_key].get('quote_count', 0)

        info = {
                'Fichier': i,  
                'ID': data['id'],
                'Nb retweet': int(retweet_count),
                'Nb like': int(like_count),
                'Nb réponses': int(reply_count),
                'Nb citations': int(quote_count),
                'Hashtags': data.get('hashtags', []),
                'Texte': str(tt),
                'Mots': tt.words,
                'Phrases': tt.sentences,
                'Tags POS': tt.tags,
                'Sentiment': tt.sentiment,
                'Polarité': tt.polarity,
                'Subjectivité': tt.subjectivity,
                'Phrases nominales': tt.noun_phrases,
                'Texte corrigé': tt.correct(),
                'Texte tokénizé': tt.tokens
        }
        data_list.append(info)
        

df = pd.DataFrame(data_list, columns=colonnes)
print(df)
df.to_csv("../analyse/covid_vaccine_analyse.csv",index=False)

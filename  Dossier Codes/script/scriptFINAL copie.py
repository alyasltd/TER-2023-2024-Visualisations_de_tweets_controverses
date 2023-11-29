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
from langdetect import detect
from multiprocessing import Pool
import time

def lire_et_parser_json(chemin):
    try:
        with open(chemin, 'r', encoding='utf-8') as fichier:
            return json.load(fichier)
    except json.JSONDecodeError as e:
        print(f"Erreur de décodage JSON : {e}")
        return None
    except FileNotFoundError as e:
        print(f"Fichier non trouvé : {e}")
        return None

def traiter_fichier(chemin_complet):
    data = lire_et_parser_json(chemin_complet)
    if data and 'text' in data and data['text'].strip():
        tt = TextBlob(data['text'])
        retweet_count = reply_count = like_count = quote_count = 0

        if 'metrics' in data and data['metrics']:
            metric_key = list(data['metrics'].keys())[0]
            retweet_count = data['metrics'][metric_key].get('retweet_count', 0)
            reply_count = data['metrics'][metric_key].get('reply_count', 0)
            like_count = data['metrics'][metric_key].get('like_count', 0)
            quote_count = data['metrics'][metric_key].get('quote_count', 0)

        return {
            'Fichier': os.path.basename(chemin_complet),  
            'ID': data['id'],
            'Langue': detect(data['text']),
            'Nb retweet': int(retweet_count),
            'Nb like': int(like_count),
            'Nb réponses': int(reply_count),
            'Nb citations': int(quote_count),
            'Hashtags': str(data.get('hashtags', [])),
            'Texte': str(tt),
            'Mots': str(tt.words),
            'Phrases': str(tt.sentences),
            'Tags POS': str(tt.tags),
            'Sentiment': str(tt.sentiment),
            'Polarité': str(tt.polarity),
            'Subjectivité': str(tt.subjectivity),
            'Phrases nominales': str(tt.noun_phrases),
            'Texte corrigé': str(tt.correct()),
            'Texte tokénizé': str(tt.tokens)
        }

if __name__ == "__main__":
    start_time = time.time()

    chemin_dossier = "../../data/tweets_covid_vaccine"
    chemins_complets = [os.path.join(chemin_dossier, i) for i in os.listdir(chemin_dossier) if i.endswith('.json')]

    data_list = []
    nombre_de_coeurs = 4
    with Pool(nombre_de_coeurs) as pool:
        data_list = pool.map(traiter_fichier, chemins_complets)

    colonnes = ['Fichier', 'ID',  'Langue', 'Nb retweet', 'Nb like', 'Nb réponses', 'Nb citations', 'Hashtags', 'Texte', 'Mots', 'Phrases', 'Tags POS', 'Sentiment', 'Polarité', 'Subjectivité', 'Phrases nominales', 'Texte corrigé', 'Texte tokénizé']
    df = pd.DataFrame(data_list, columns=colonnes)
    
    end_time = time.time()
    print(f"Temps total d'exécution : {end_time - start_time} secondes")
    
    df.to_csv("../analyse_Audric2/covid_vaccine_analyse.csv", index=False)

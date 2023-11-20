"""
ce script est utilisé pour filtrer et copier des fichiers JSON (représentant des tweets) 
d'un dossier source à un dossier cible en fonction de critères spécifiques (présence de 
texte dans le champ 'text' du JSON). Il s'appuie sur un ensemble d'identifiants de tweets 
fournis dans un fichier CSV pour déterminer quels fichiers doivent être traités et copiés.
il faut adapter le dossier cible et le df2 en fonction de ce que l'on veut.
"""

import os
import shutil
import json
import pandas as pd
from textblob import TextBlob

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

chemin_dossier_source = "../../data/tweets/"
chemin_dossier_cible = "../../data/tweets_covid_vaccine/"
df2 = pd.read_csv("../INM/#covid19_#vaccine/covid_vaccine_json.csv")
id_set = set(df2['0'])

# Créer le dossier cible s'il n'existe pas
if not os.path.exists(chemin_dossier_cible):
    os.makedirs(chemin_dossier_cible)

# Traiter et copier les fichiers
for nom_fichier in (id_set):
        chemin_complet = os.path.join(chemin_dossier_source, nom_fichier)
        data = lire_et_parser_json(chemin_complet)
        if data and 'text' in data and data['text'].strip():
            chemin_complet_cible = os.path.join(chemin_dossier_cible, nom_fichier)
            
            # Copier le fichier si du texte est présent
            shutil.copy(chemin_complet, chemin_complet_cible)
            print(f"Fichier copié : {nom_fichier}")

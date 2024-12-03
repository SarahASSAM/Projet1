import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Chemin du fichier JSONL
file_path = "reviews.jsonl"

# Lire les 200 premières lignes
data = []
with open(file_path, "r", encoding="utf-8") as file:
    for i, line in enumerate(file):
        if i >= 200:  # Limiter à 200 avis
            break
        data.append(json.loads(line.strip()))  # Charger chaque ligne comme un objet JSON

# Extraire les notes et les textes
ratings = [review.get("rating") for review in data]
texts = [review.get("text") for review in data]

# Convertir en DataFrame
reviews_df = pd.DataFrame({"rating": ratings, "text": texts})

# Afficher un aperçu des données
print("Aperçu des données chargées :")
print(reviews_df.head())

# Partie 2 : Chargement du modèle et analyse des sentiments
# Charger le tokenizer et le modèle
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Fonction pour analyser les sentiments d'une liste de textes
def analyze_sentiments(texts):
    predictions = []
    for text in texts:
        # Préparer les entrées
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        # Obtenir les prédictions
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item() + 1  # Les classes commencent à 1
        predictions.append(predicted_class)
    return predictions

# Appliquer l'analyse des sentiments aux 200 avis
predicted_ratings = analyze_sentiments(reviews_df["text"])

# Ajouter les prédictions au DataFrame
reviews_df["predicted_rating"] = predicted_ratings

# Afficher un aperçu des résultats
print("Aperçu des résultats après analyse des sentiments :")
print(reviews_df.head())

# Sauvegarder les résultats dans un fichier CSV
reviews_df.to_csv("reviews_with_sentiments.csv", index=False)
print("Les résultats ont été sauvegardés dans 'reviews_with_sentiments.csv'.")

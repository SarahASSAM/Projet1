import pandas as pd
import json

# Chemin du fichier JSONL
file_path = "reviews.jsonl"

# Lire les données ligne par ligne
data = []
with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        data.append(json.loads(line.strip()))  # Charger chaque ligne comme un objet JSON

# Extraire les notes et les textes
ratings = [review.get("rating") for review in data]
texts = [review.get("text") for review in data]

# Convertir en DataFrame
reviews_df = pd.DataFrame({"rating": ratings, "text": texts})

# Afficher un aperçu des données
print(reviews_df.head())
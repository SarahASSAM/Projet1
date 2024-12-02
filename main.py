import jsonlines
import pandas as pd

# Chemin vers le fichier
reviews_file = r"C:\Users\sarah\Desktop\Cours M2\NLP & GEN AI\reviews.jsonl"

# Charger les avis clients
reviews = []
with jsonlines.open(reviews_file) as reader:
    for obj in reader:
        reviews.append(obj)

# Convertir en DataFrame
df = pd.DataFrame(reviews)

# Garder uniquement les champs pertinents (title et text)
df = df[['title', 'text']]

# Vérifier les premières lignes
print("Aperçu des données :")
print(df.head())


import jsonlines
import pandas as pd
import spacy

# Charger les données
reviews_file = r"C:\Users\sarah\Desktop\Cours M2\NLP & GEN\reviews.jsonl"

# Charger les avis clients
reviews = []
with jsonlines.open(reviews_file) as reader:
    for obj in reader:
        reviews.append(obj)

# Convertir en DataFrame
df_reviews = pd.DataFrame(reviews)

# Garder uniquement les champs pertinents
df_reviews = df_reviews[['title', 'text']]

# Combiner 'title' et 'text' pour créer une colonne 'document'
df_reviews['document'] = df_reviews['title'] + " " + df_reviews['text']

# Charger le modèle SpaCy
nlp = spacy.load("en_core_web_sm")

# Fonction de tokenisation
def tokenize_text(text):
    doc = nlp(text)
    return [token.text for token in doc]  # Liste des tokens

# Appliquer la tokenisation à chaque document
df_reviews['tokens'] = df_reviews['document'].apply(tokenize_text)

# Afficher les résultats
print("Exemple de tokenisation :")
print(df_reviews[['document', 'tokens']].head())

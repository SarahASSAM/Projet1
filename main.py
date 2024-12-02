import jsonlines
import pandas as pd
import spacy
import json
import re
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

# Convertir le DataFrame en format JSON pour sauvegarde
tokenized_data = df_reviews[['document', 'tokens']].to_dict(orient='records')

# Chemin de sauvegarde
output_file = r"C:\Users\sarah\Desktop\Cours M2\NLP & GEN\tokenized_reviews.json"

# Sauvegarder en JSON
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(tokenized_data, f, ensure_ascii=False, indent=4)

print(f"Tokens sauvegardés dans : {output_file}")

# Fonction pour effectuer la lemmatisation
def lemmatize_text(text):
    doc = nlp(text)  # Processus NLP avec SpaCy
    return [token.lemma_ for token in doc]  # Liste des lemmes
# Appliquer la lemmatisation sur chaque document
df_reviews['lemmas'] = df_reviews['document'].apply(lemmatize_text)

# Afficher un aperçu des lemmes générés
print("Exemple de lemmatisation :")
print(df_reviews[['document', 'lemmas']].head())
# Fonction pour supprimer les stop words
def remove_stop_words(tokens):
    return [token for token in tokens if not nlp.vocab[token].is_stop]

# Appliquer la suppression des stop words sur la liste des lemmes
df_reviews['filtered'] = df_reviews['lemmas'].apply(remove_stop_words)

# Afficher un aperçu des données après suppression des stop words
print("Exemple après suppression des stop words :")
print(df_reviews[['lemmas', 'filtered']].head())
# Fonction pour exclure les éléments non pertinents
def clean_tokens(tokens):
    return [
        token for token in tokens
        if token.isalpha()  # Garde uniquement les mots (exclut les chiffres et la ponctuation)
        and not re.match(r'^www\.|https?://', token)  # Exclut les URLs
        ]
# Appliquer le nettoyage sur les tokens filtrés
df_reviews['cleaned'] = df_reviews['filtered'].apply(clean_tokens)

# Afficher un aperçu des données après nettoyage
print("Exemple après nettoyage :")
print(df_reviews[['filtered', 'cleaned']].head())

# Sauvegarder les données nettoyées dans un fichier JSON
prepared_data = df_reviews[['document', 'cleaned']].to_dict(orient='records')

# Chemin de sauvegarde pour les données nettoyées
final_output_file = r"C:\Users\sarah\Desktop\Cours M2\NLP & GEN\prepared_reviews.json"

# Sauvegarder en JSON
with open(final_output_file, 'w', encoding='utf-8') as f:
    json.dump(prepared_data, f, ensure_ascii=False, indent=4)

print(f"Données nettoyées sauvegardées dans : {final_output_file}")


# Chemin vers le fichier JSON
input_file = r"C:\Users\sarah\Desktop\Cours M2\NLP & GEN\prepared_reviews.json"

# Charger les données prétraitées
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extraire les tokens nettoyés
documents = [" ".join(doc['cleaned']) for doc in data]

# Afficher un aperçu des documents
print("Exemple de documents :")
print(documents[:3])

from sentence_transformers import SentenceTransformer

# Charger un modèle pré-entraîné
model = SentenceTransformer('all-MiniLM-L6-v2')

# Générer les embeddings pour chaque document nettoyé
embeddings = model.encode(documents)

# Afficher la forme des embeddings
print(f"Shape des embeddings : {len(embeddings)}, {len(embeddings[0])}")

# Exemple d'un embedding
print("Exemple d'embedding (vecteur) :", embeddings[0])

# Afficher les 3 premiers vecteurs
print("Les trois premiers embeddings :")
for i in range(3):
    print(f"Embedding {i + 1} : {embeddings[i]}")

# Sauvegarder les embeddings avec leurs documents
embeddings_data = [{"document": documents[i], "embedding": embeddings[i].tolist()} for i in range(len(embeddings))]

# Chemin de sauvegarde
embeddings_file = r"C:\Users\sarah\Desktop\Cours M2\NLP & GEN\embeddings.json"

# Écrire dans un fichier JSON
with open(embeddings_file, 'w', encoding='utf-8') as f:
    json.dump(embeddings_data, f, ensure_ascii=False, indent=4)

print(f"Embeddings sauvegardés dans : {embeddings_file}")


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialiser le vectoriseur TF-IDF
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')  # Limiter à 1000 mots les plus fréquents

# Générer la matrice TF-IDF
tfidf_matrix = vectorizer.fit_transform(documents)

# Extraire les termes (mots) associés aux colonnes de la matrice
terms = vectorizer.get_feature_names_out()

# Convertir la matrice sparse en format dense
dense_tfidf = tfidf_matrix.todense()

# Convertir en liste de dictionnaires
tfidf_data = []
for i, doc in enumerate(dense_tfidf):
    tfidf_data.append({
        "document": documents[i],
        "tfidf_scores": {terms[j]: doc[0, j] for j in range(len(terms)) if doc[0, j] > 0}  # Enregistrer uniquement les scores non nuls
    })

# Chemin pour sauvegarder le fichier JSON
output_file = r"C:\Users\sarah\Desktop\Cours M2\NLP & GEN\tfidf_reviews.json"

# Sauvegarder en JSON
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(tfidf_data, f, ensure_ascii=False, indent=4)

print(f"Matrice TF-IDF sauvegardée sous forme JSON dans : {output_file}")

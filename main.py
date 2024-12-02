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



from sklearn.feature_extraction.text import TfidfVectorizer

# Initialiser le vectoriseur TF-IDF
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')  # Limiter à 1000 mots les plus fréquents

# Générer la matrice TF-IDF sous forme sparse
tfidf_sparse_matrix = vectorizer.fit_transform(documents)

# Afficher des informations sur la matrice sparse
print("Matrice TF-IDF (sparse) :")
print(tfidf_sparse_matrix)

# Dimensions de la matrice TF-IDF
print(f"Dimensions de la matrice sparse TF-IDF : {tfidf_sparse_matrix.shape}")

# Afficher un échantillon des données sparse
print("Aperçu des données non nulles dans la matrice sparse TF-IDF :")
print(tfidf_sparse_matrix[:3, :])  # Affiche les 3 premières lignes

# Sauvegarder la matrice sparse dans un fichier
from scipy.sparse import save_npz

sparse_output_file = r"C:\Users\sarah\Desktop\Cours M2\NLP & GEN\tfidf_sparse_matrix.npz"

# Sauvegarder au format .npz
save_npz(sparse_output_file, tfidf_sparse_matrix)

print(f"Matrice TF-IDF sparse sauvegardée dans : {sparse_output_file}")





from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Calculer une matrice de similarité cosinus à partir de la matrice TF-IDF sparse
cosine_sim_matrix = cosine_similarity(tfidf_sparse_matrix)

# Convertir la similarité cosinus en matrice de distances
distance_matrix = 1 - cosine_sim_matrix

# Corriger les éventuelles valeurs négatives dues à des erreurs d'arrondi
distance_matrix = np.clip(distance_matrix, 0, None)

# Définir l'algorithme DBSCAN avec une distance epsilon (paramètre à ajuster selon les données)
dbscan = DBSCAN(eps=0.5, min_samples=5, metric='precomputed')

# Appliquer DBSCAN sur la matrice de distances
clusters = dbscan.fit_predict(distance_matrix)

# Ajouter les étiquettes de cluster dans le DataFrame
df_reviews['cluster'] = clusters

# Afficher un aperçu des clusters générés
print("Aperçu des clusters générés :")
print(df_reviews[['document', 'cluster']].head(10))

# Vérifier le nombre de clusters trouvés
num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)  # -1 représente les points "bruit" non assignés
print(f"Nombre de clusters identifiés (hors bruit) : {num_clusters}")

# Sauvegarder les résultats dans un fichier JSON
clustering_results = df_reviews[['document', 'cluster']].to_dict(orient='records')
clustering_output_file = r"C:\Users\sarah\Desktop\Cours M2\NLP & GEN\clustering_results.json"

with open(clustering_output_file, 'w', encoding='utf-8') as f:
    json.dump(clustering_results, f, ensure_ascii=False, indent=4)

print(f"Résultats de clustering sauvegardés dans : {clustering_output_file}")

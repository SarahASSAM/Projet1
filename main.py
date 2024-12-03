import jsonlines
import pandas as pd
import spacy
import json
import re
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import save_npz

# Charger les données
reviews_file = "reviews.jsonl"
reviews = [obj for obj in jsonlines.open(reviews_file)]

# Préparer le DataFrame
df_reviews = pd.DataFrame(reviews)[['title', 'text']]
df_reviews['document'] = df_reviews['title'] + " " + df_reviews['text']

# Charger le modèle SpaCy
nlp = spacy.load("en_core_web_sm")

# Pipeline de prétraitement
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    lemmas = [token.lemma_ for token in doc]
    filtered = [lemma for lemma in lemmas if not nlp.vocab[lemma].is_stop]
    cleaned = [token for token in filtered if token.isalpha()]
    return tokens, lemmas, filtered, cleaned

df_reviews[['tokens', 'lemmas', 'filtered', 'cleaned']] = df_reviews['document'].apply(
    lambda x: pd.Series(preprocess_text(x))
)

# Sauvegarder les données nettoyées
cleaned_data = df_reviews[['document', 'cleaned']].to_dict(orient='records')
with open("prepared_reviews.json", 'w', encoding='utf-8') as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=4)

# Génération des embeddings
documents = [" ".join(doc['cleaned']) for doc in cleaned_data]
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)

# Sauvegarder les embeddings
embeddings_data = [{"document": documents[i], "embedding": embeddings[i].tolist()} for i in range(len(embeddings))]
with open("embeddings.json", 'w', encoding='utf-8') as f:
    json.dump(embeddings_data, f, ensure_ascii=False, indent=4)

# TF-IDF Vectorisation
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_sparse_matrix = vectorizer.fit_transform(documents)
save_npz("tfidf_sparse_matrix.npz", tfidf_sparse_matrix)

# DBSCAN Clustering
cosine_sim_matrix = cosine_similarity(tfidf_sparse_matrix)
distance_matrix = np.clip(1 - cosine_sim_matrix, 0, None)
dbscan = DBSCAN(eps=0.5, min_samples=5, metric='precomputed')
clusters = dbscan.fit_predict(distance_matrix)
df_reviews['cluster'] = clusters


# Vérification des clusters générés
print("Clusters identifiés :", np.unique(clusters))

# Analyse des clusters
clustered_documents = df_reviews.groupby('cluster')['cleaned'].apply(list).to_dict()

# Préparer les documents et clusters pour TF-IDF
documents_with_cluster = [
    (" ".join(tokens), cluster) for cluster, tokens_list in clustered_documents.items() for tokens in tokens_list
]
texts, labels = zip(*documents_with_cluster)

# Vectorisation TF-IDF par classe
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
tfidf_vocab = tfidf_vectorizer.get_feature_names_out()

# Obtenir les mots et scores TF-IDF par cluster
cluster_words = {}
for cluster in set(labels):
    cluster_indices = [i for i, lbl in enumerate(labels) if lbl == cluster]
    cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0).A1
    cluster_words[cluster] = sorted(
        zip(tfidf_vocab, cluster_tfidf),
        key=lambda x: x[1],
        reverse=True
    )[:10]

# Afficher les mots les plus pertinents par cluster
for cluster, words in cluster_words.items():
    print(f"Cluster {cluster} (TF-IDF) :")
    for word, score in words:
        print(f"  {word} : {score:.4f}")
    print("\n")

# Extraction des n-grams
def generate_ngrams(documents, n=2):
    vectorizer = CountVectorizer(ngram_range=(n, n), stop_words='english')
    ngram_matrix = vectorizer.fit_transform(documents)
    ngram_counts = ngram_matrix.sum(axis=0)
    ngram_vocab = vectorizer.get_feature_names_out()
    return Counter(dict(zip(ngram_vocab, ngram_counts.A1)))

bigrams_per_cluster = {
    cluster: generate_ngrams([" ".join(tokens) for tokens in tokens_list], n=2).most_common(10)
    for cluster, tokens_list in clustered_documents.items()
}

# Afficher les bigrams principaux par cluster
for cluster, bigrams in bigrams_per_cluster.items():
    print(f"Cluster {cluster} (Bigrams) :")
    for bigram, freq in bigrams:
        print(f"  {bigram} : {freq}")
    print("\n")

# Visualisation avec t-SNE
if len(set(clusters)) > 1:
    tsne_data = TSNE(n_components=2, random_state=42).fit_transform(tfidf_sparse_matrix.toarray())
    plt.figure(figsize=(12, 8))
    for cluster in set(clusters):
        cluster_points = tsne_data[np.where(clusters == cluster)]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}" if cluster != -1 else "Bruit")
    plt.title("Clusters DBSCAN avec t-SNE")
    plt.legend()
    plt.show()

# Réduction de dimension avec PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(tfidf_sparse_matrix.toarray())

# Préparer la visualisation
plt.figure(figsize=(12, 8))
for cluster in set(clusters):
    cluster_points = pca_data[np.where(clusters == cluster)]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}" if cluster != -1 else "Bruit")
plt.title("Clusters DBSCAN avec PCA (Projection 2D)")
plt.legend()
plt.show()

# Sauvegarder les clusters avec documents associés
final_clusters = df_reviews.groupby('cluster')['document'].apply(list).to_dict()
with open("clusters_with_documents.json", 'w', encoding='utf-8') as f:
    json.dump(final_clusters, f, ensure_ascii=False, indent=4)

# Extraction des entités nommées (NER)
def extract_entities(documents):
    # Convertir les documents en chaînes de caractères si nécessaire
    documents = [" ".join(doc) if isinstance(doc, list) else doc for doc in documents]
    entities = []
    for doc in nlp.pipe(documents, batch_size=50):  # Traitement en lot pour accélérer
        entities.extend([ent.text for ent in doc.ents])
    return entities

ner_per_cluster = {
    cluster: Counter(extract_entities(docs)).most_common(10)
    for cluster, docs in clustered_documents.items()
}

# Sauvegarder les entités nommées pour chaque cluster
with open("ner_per_cluster.json", 'w', encoding='utf-8') as f:
    json.dump(ner_per_cluster, f, ensure_ascii=False, indent=4)

# Sauvegarder les fréquences des mots pour chaque cluster
word_frequencies = {
    cluster: Counter([token for tokens in tokens_list for token in tokens])
    for cluster, tokens_list in clustered_documents.items()
}
with open("word_frequencies.json", 'w', encoding='utf-8') as f:
    json.dump(
        {cluster: dict(freq.most_common(10)) for cluster, freq in word_frequencies.items()},
        f, ensure_ascii=False, indent=4
    )

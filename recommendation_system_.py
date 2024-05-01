# -*- coding: utf-8 -*-
"""
@author: kieu_
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


movies = pd.read_csv(r"C:\Users\kieu_\OneDrive\Desktop\Project\Data Science\ML_9-1\Datasets\movie_data\movies.csv", encoding='latin-1', sep="\t", usecols=["movie_id", "title", "genres"])
movies.head()


# Split les valeurs dans 'genres'
def clean_genres(genres):
    result = " ".join(genres.replace("-", "").split("|")) # replace("-", "") permet de ne pas séparer le mot 'sci-fi'
    return result

movies["genres"] = movies["genres"].apply(clean_genres)



tf = TfidfVectorizer()
tfidf_matrix = tf.fit_transform(movies["genres"])


print(tfidf_matrix.shape)
# (3883, 18)

print(len(tf.vocabulary_))
# 18

print(tf.vocabulary_)
# {'animation': 2, 'children': 3, 'comedy': 4, 'adventure': 1, 'fantasy': 8, 'romance': 13,
# 'drama': 7, 'action': 0, 'crime': 5, 'thriller': 15, 'horror': 10, 'scifi': 14,
# 'documentary': 6, 'war': 16, 'musical': 11, 'mystery': 12, 'filmnoir': 9, 'western': 17}


# Afficher la matrice
tfidf_matrix_dense = pd.DataFrame(tfidf_matrix.todense(), columns=tf.get_feature_names(), index=movies["title"])


# Calculer la similiarité entre les exemples (films) dans X et Y
cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim, columns=movies["title"], index=movies["title"])


# Fonction de recommandation de films similaires
def get_recommendations(title, top_k, df):
    data = df.loc[title, :]
    data = data.sort_values(ascending=False)
    return data[:top_k].to_frame(name="score")


# Exemple 1
title = "Leaving Las Vegas (1995)"
top_k = 20
result = get_recommendations(title, top_k, cosine_sim_df)

# Exemple 2
title = "Indian in the Cupboard, The (1995)"
top_k = 20
result = get_recommendations(title, top_k, cosine_sim_df)


    
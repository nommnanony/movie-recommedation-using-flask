import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load dataset
movies = pd.read_csv("movies.csv")

# Keep only required columns
movies = movies[['movieId', 'title', 'genres']]

# Preprocess text
movies['genres'] = movies['genres'].str.replace('|', ' ', regex=True)
movies['tags'] = movies['title'] + " " + movies['genres']

# Feature extraction
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# Similarity matrix
similarity = cosine_similarity(vectors)

# Create models folder
os.makedirs("models", exist_ok=True)

# Save pickles
pickle.dump(movies, open("models/movies.pkl", "wb"))
pickle.dump(similarity, open("models/similarity.pkl", "wb"))

print("✅ Pickle files created: models/movies.pkl, models/similarity.pkl")

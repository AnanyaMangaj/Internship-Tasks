import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path

# Load data
books_path = Path("data/books.csv")
if not books_path.exists():
    books_path = Path("books.csv")

books = pd.read_csv(books_path)

# Since we don't have ratings data, create similarity based on genre and author
books['features'] = books['genre'] + " " + books['author']
books['features'] = books['features'].fillna("").str.lower()

# Vectorization
cv = CountVectorizer(max_features=100, stop_words='english')
vectors = cv.fit_transform(books['features']).toarray()

# Similarity
similarity = cosine_similarity(vectors)

# Save
Path("model").mkdir(parents=True, exist_ok=True)
with open("model/books.pkl", "wb") as f:
    pickle.dump(books, f)
with open("model/book_similarity.pkl", "wb") as f:
    pickle.dump(similarity, f)

print("Book model built successfully")

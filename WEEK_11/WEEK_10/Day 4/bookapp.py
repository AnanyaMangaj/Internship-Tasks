import streamlit as st
import pickle

# Load model
books = pickle.load(open("model/books.pkl", "rb"))
similarity = pickle.load(open("model/book_similarity.pkl", "rb"))

st.set_page_config(page_title="Book Recommender", layout="wide")

st.title("📚 Book Recommendation System")

# Recommend function
def recommend(book_title):
    book_title = book_title.lower().strip()

    matches = books[books['title'].str.lower() == book_title]

    if matches.empty:
        return ["❌ Book not found"]

    book_index = matches.index[0]
    distances = similarity[book_index]

    books_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:10]

    return [books.iloc[i[0]].title for i in books_list]

# UI
book_list = books['title'].tolist()

selected_book = st.selectbox("Select a Book", book_list)

if st.button("Recommend"):
    recommendations = recommend(selected_book)

    for book in recommendations:
        st.write("👉", book)

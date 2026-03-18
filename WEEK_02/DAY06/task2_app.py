import streamlit as st

st.title("Programming Language Selector")

language = st.selectbox(
    "Choose a programming language:",
    ["Python", "Java", "C++", "JavaScript", "Go"]
)

show = st.checkbox("Show Selected Language")

if show:
    st.success(f"You selected: {language}")

import streamlit as st

st.title("User Skill App")

if "users" not in st.session_state:
    st.session_state.users = []

st.header("Add User")

name = st.text_input("Enter Name")
skills = st.text_input("Enter Skills (comma separated)")

if st.button("Add Details"):
    if name and skills:
        st.session_state.users.append({
            "name": name,
            "skills": skills
        })
        st.success("Details Added Successfully")
    else:
        st.error("Please enter both name and skills")

st.divider()

st.header("Search User")

search_name = st.text_input("Enter Name to Search")
search_age = st.number_input("Enter Age", min_value=1, max_value=100, step=1)

if st.button("Search"):
    found = False
    for user in st.session_state.users:
        if user["name"] == search_name:
            st.success("User Found")
            st.write("Name:", user["name"])
            st.write("Skills:", user["skills"])
            st.write("Age Entered:", search_age)
            found = True
            break

    if not found:
        st.error("User not found")

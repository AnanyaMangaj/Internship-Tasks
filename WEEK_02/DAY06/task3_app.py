import streamlit as st

st.title("Visible Counter App")

limit = 5

if "count" not in st.session_state:
    st.session_state.count = 0

st.write("Counter Value:", st.session_state.count)

if st.button("Increase Counter"):
    st.session_state.count += 1

    if st.session_state.count >= limit:
        st.session_state.count = 0
        st.warning("Limit reached! Counter reset to 0")

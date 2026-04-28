import streamlit as st

st.set_page_config(layout="wide")

pages = st.navigation([
    st.Page("pages/free-search-page.py", title="Search", icon="🔍"),
    st.Page("pages/similarity-page.py",  title="Movie similarity",  icon="🎬"),
])
pages.run()

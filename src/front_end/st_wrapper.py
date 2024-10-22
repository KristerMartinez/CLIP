import streamlit as st
from src.front_end.core.config import settings
from streamlit_navigation_bar import st_navbar
import src.front_end.pages as pg

def create_streamlit():

    options = {
    "show_menu": False,
    "show_sidebar": False,
    }
    

    st.title(settings.NAVBAR['Title'])
    # sl = create_sidebar(sl)

    # Create the links for the navbar
    pages = [link['title'] for link in settings.NAVBAR['Links']]

    # Push the title to the front of the list
    pages.insert(0, settings.NAVBAR['Title'])
    print(pages) 

    page = st_navbar(
        pages,
        # logo_path=logo_path,
        # urls=urls,
        # styles=styles,
        options=options,
)

    
    functions = {
    'Project CAI 2840C': pg.show_dashboard,
    'Dashboard': pg.show_dashboard,
    'Use Case 1': pg.show_usecase1,
    'About': pg.show_about,
}
    go_to = functions.get(page)
    if go_to:
        go_to()


# def create_navbar(sl, links):
#     for link in links:
#         sl.write(link)
#     page = st_navbar(links)
#     sl.write(page)

#     return sl



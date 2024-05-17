##### NEXT Step: Build Knowledge Graph with LLM. Maybe use Langchain?

# Standard library imports
import streamlit as st
import datetime
import base64
import concurrent.futures
import time

# Related third-party imports
from streamlit_elements import Elements
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import pandas as pd
import searchconsole
from stqdm import stqdm
import nltk
from nltk.corpus import stopwords
from stqdm import stqdm

# LLM imports
from anthropic import Anthropic
from openai import OpenAI
from groq import Groq

import networkx as nx
import community as community_louvain
from typing import Dict, Set
import Levenshtein
from rake_nltk import Rake
import spacy
from spacy_entity_linker import EntityLinker

# -------------
# Constants
# -------------
SEARCH_TYPE = "web"
DATE_RANGE_OPTIONS = [
    "Last 7 Days",
    "Last 30 Days",
    "Last 3 Months",
    "Last 6 Months",
    "Last 12 Months",
    "Last 16 Months",
]
DEVICE_OPTIONS = ["All Devices", "desktop", "mobile", "tablet"]
DIMENSIONS = ["page", "query"]
MAX_ROWS = 250_000
DF_PREVIEW_ROWS = 100
MAX_POSITION = 5

# Define models
ANTHROPIC_MODELS = ['claude-3-opus-20240229', 'claude-3-sonnet-20240229','claude-3-haiku-20240307']
GROQ_MODELS = ['llama3-70b-8192', 'mixtral-8x7b-32768']
OPENAI_MODELS = ['gpt-4-turbo', 'gpt-3.5-turbo']
MODELS = GROQ_MODELS + ANTHROPIC_MODELS + OPENAI_MODELS
TEMPERATURE = 0.2
MAX_TOKEN = 10
LANGUAGES = ["Afrikaans","Albanian","Amharic","Arabic","Armenian","Azerbaijani","Basque","Belarusian","Bengali","Bosnian","Bulgarian","Catalan","Cebuano","Chinese (Simplified)","Chinese (Traditional)","Corsican","Croatian","Czech","Danish","Dutch","English","Esperanto","Estonian","Finnish","French","Frisian","Galician","Georgian","German","Greek","Gujarati","Haitian Creole","Hausa","Hawaiian","Hebrew","Hindi","Hmong","Hungarian","Icelandic","Igbo","Indonesian","Irish","Italian","Japanese","Javanese","Kannada","Kazakh","Khmer","Kinyarwanda","Korean","Kurdish","Kyrgyz","Lao","Latvian","Lithuanian","Luxembourgish","Macedonian","Malagasy","Malay","Malayalam","Maltese","Maori","Marathi","Mongolian","Myanmar (Burmese)","Nepali","Norwegian","Nyanja (Chichewa)","Odia (Oriya)","Pashto","Persian","Polish","Portuguese (Portugal","Punjabi","Romanian","Russian","Samoan","Scots Gaelic","Serbian","Sesotho","Shona","Sindhi","Sinhala (Sinhalese)","Slovak","Slovenian","Somali","Spanish","Sundanese","Swahili","Swedish","Tagalog (Filipino)","Tajik","Tamil","Tatar","Telugu","Thai","Turkish","Turkmen","Ukrainian","Urdu","Uyghur","Uzbek","Vietnamese","Welsh","Xhosa","Yiddish","Yoruba","Zulu"]
COUNTRIES = ["Afghanistan", "Albania", "Antarctica", "Algeria", "American Samoa", "Andorra", "Angola", "Antigua and Barbuda", "Azerbaijan", "Argentina", "Australia", "Austria", "The Bahamas", "Bahrain", "Bangladesh", "Armenia", "Barbados", "Belgium", "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Belize", "Solomon Islands", "Brunei", "Bulgaria", "Myanmar (Burma)", "Burundi", "Cambodia", "Cameroon", "Canada", "Cape Verde", "Central African Republic", "Sri Lanka", "Chad", "Chile", "China", "Christmas Island", "Cocos (Keeling) Islands", "Colombia", "Comoros", "Republic of the Congo", "Democratic Republic of the Congo", "Cook Islands", "Costa Rica", "Croatia", "Cyprus", "Czechia", "Benin", "Denmark", "Dominica", "Dominican Republic", "Ecuador", "El Salvador", "Equatorial Guinea", "Ethiopia", "Eritrea", "Estonia", "South Georgia and the South Sandwich Islands", "Fiji", "Finland", "France", "French Polynesia", "French Southern and Antarctic Lands", "Djibouti", "Gabon", "Georgia", "The Gambia", "Germany", "Ghana", "Kiribati", "Greece", "Grenada", "Guam", "Guatemala", "Guinea", "Guyana", "Haiti", "Heard Island and McDonald Islands", "Vatican City", "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iraq", "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Kazakhstan", "Jordan", "Kenya", "South Korea", "Kuwait", "Kyrgyzstan", "Laos", "Lebanon", "Lesotho", "Latvia", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Mauritania", "Mauritius", "Mexico", "Monaco", "Mongolia", "Moldova", "Montenegro", "Morocco", "Mozambique", "Oman", "Namibia", "Nauru", "Nepal", "Netherlands", "Curacao", "Sint Maarten", "Caribbean Netherlands", "New Caledonia", "Vanuatu", "New Zealand", "Nicaragua", "Niger", "Nigeria", "Niue", "Norfolk Island", "Norway", "Northern Mariana Islands", "United States Minor Outlying Islands", "Federated States of Micronesia", "Marshall Islands", "Palau", "Pakistan", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Pitcairn Islands", "Poland", "Portugal", "Guinea-Bissau", "Timor-Leste", "Qatar", "Romania", "Rwanda", "Saint Helena, Ascension and Tristan da Cunha", "Saint Kitts and Nevis", "Saint Lucia", "Saint Pierre and Miquelon", "Saint Vincent and the Grenadines", "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Vietnam", "Slovenia", "Somalia", "South Africa", "Zimbabwe", "Spain", "Suriname", "Eswatini", "Sweden", "Switzerland", "Tajikistan", "Thailand", "Togo", "Tokelau", "Tonga", "Trinidad and Tobago", "United Arab Emirates", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "North Macedonia", "Egypt", "United Kingdom", "Guernsey", "Jersey", "Tanzania", "United States", "Burkina Faso", "Uruguay", "Uzbekistan", "Venezuela", "Wallis and Futuna", "Samoa", "Yemen", "Zambia"]

# -------------
# Variables
# -------------
preferred_countries = ["Germany", "Austria", "Switzerland", "United Kingdom", "United States", "France", "Italy", "Netherlands"]
preferred_languages = ["German", "English", "French", "Italian", "Dutch"]

# -------------
# Streamlit App Configuration
# -------------

def setup_streamlit():
    """
    Configures Streamlit's page settings and displays the app title and markdown information.
    Sets the page layout, title, and markdown content with links and app description.
    """
    st.set_page_config(
        page_title="Topical Authority Analysis",
        page_icon=":crown:",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.linkedin.com/in/kirchhoff-kevin/',
            'About': "This is an app for generating semantic sitemaps and analyzing topical authority!"
        }
    )
    st.image("https://www.claneo.com/wp-content/uploads/Element-4.svg", width=600, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    st.caption("👋 Developed by [Kevin](https://www.linkedin.com/in/kirchhoff-kevin/)")
    st.title("Topical Authority Analysis")
    st.divider()

def init_session_state():
    """
    Initialises or updates the Streamlit session state variables for property selection,
    search type, date range, dimensions, and device type.
    """
    if 'selected_property' not in st.session_state:
        st.session_state.selected_property = None
    if 'selected_search_type' not in st.session_state:
        st.session_state.selected_search_type = 'web'
    if 'selected_date_range' not in st.session_state:
        st.session_state.selected_date_range = 'Last 6 Months'
    if 'selected_dimensions' not in st.session_state:
        st.session_state.selected_dimensions = ['page', 'query']
    if 'selected_device' not in st.session_state:
        st.session_state.selected_device = 'All Devices'
    if 'selected_min_clicks' not in st.session_state:
        st.session_state.selected_min_clicks = 100


# -------------
# Google Authentication Functions
# -------------

def load_config():
    """
    Loads the Google API client configuration from Streamlit secrets.
    Returns a dictionary with the client configuration for OAuth.
    """
    client_config = {
        "installed": {
            "client_id": str(st.secrets["installed"]["client_id"]),
            "client_secret": str(st.secrets["installed"]["client_secret"]),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://accounts.google.com/o/oauth2/token",
            "redirect_uris": (
                [str(st.secrets["installed"]["redirect_uris"][0])]
            ),
        }
    }
    return client_config


def init_oauth_flow(client_config):
    """
    Initialises the OAuth flow for Google API authentication using the client configuration.
    Sets the necessary scopes and returns the configured Flow object.
    """
    scopes = ["https://www.googleapis.com/auth/webmasters"]
    return Flow.from_client_config(
        client_config,
        scopes=scopes,
        redirect_uri=client_config["installed"]["redirect_uris"][0],
    )


def google_auth(client_config):
    """
    Starts the Google authentication process using OAuth.
    Generates and returns the OAuth flow and the authentication URL.
    """
    flow = init_oauth_flow(client_config)
    auth_url, _ = flow.authorization_url(prompt="consent")
    return flow, auth_url


def auth_search_console(client_config, credentials):
    """
    Authenticates the user with the Google Search Console API using provided credentials.
    Returns an authenticated searchconsole client.
    """
    token = {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_uri": credentials.token_uri,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes": credentials.scopes,
        "id_token": getattr(credentials, "id_token", None),
    }
    return searchconsole.authenticate(client_config=client_config, credentials=token)


# -------------
# Data Fetching Functions
# -------------

def list_gsc_properties(credentials):
    """
    Lists all Google Search Console properties accessible with the given credentials.
    Returns a list of property URLs or a message if no properties are found.
    """
    service = build('webmasters', 'v3', credentials=credentials)
    site_list = service.sites().list().execute()
    return [site['siteUrl'] for site in site_list.get('siteEntry', [])] or ["No properties found"]

def fetch_gsc_data(webproperty, search_type, start_date, end_date, dimensions, min_clicks, directory, device_type=None):
    """
    Fetches Google Search Console data for a specified property, date range, dimensions, and device type.
    Handles errors and returns the data as a DataFrame.
    """
    query = webproperty.query.range(start_date, end_date).search_type(search_type).dimension(*dimensions)

    try:
        df = query.limit(MAX_ROWS).get().to_dataframe()
        
        if directory:
            # Filter URLs based on the provided directory using regex
            df = df[df['page'].str.contains(f".*{directory}.*", regex=True)]
        
        if 'clicks' in df.columns and 'position' in df.columns:
            if min_clicks is not None and min_clicks > 0:
                df = df[df['clicks'] >= min_clicks]
                df = df[df['position'] <= MAX_POSITION]
            else:
                st.write("Skipping filtering based on clicks as min_clicks is None or 0.")
        else:
            show_error("Columns 'clicks' or 'position' not in DataFrame")
        
        return df
    except Exception as e:
        show_error(e)
        return pd.DataFrame()


def fetch_data_loading(webproperty, search_type, start_date, end_date, dimensions, min_clicks, directory, device_type=None):
    """
    Fetches Google Search Console data with a loading indicator. Utilises 'fetch_gsc_data' for data retrieval.
    Returns the fetched data as a DataFrame.
    """
    with st.spinner('Fetching data...'):
        return fetch_gsc_data(webproperty, search_type, start_date, end_date, dimensions, min_clicks, directory, device_type)


# -------------
# Utility Functions
# -------------
def custom_sort(all_items, preferred_items):
    sorted_items = preferred_items + ["_____________"] + [item for item in all_items if item not in preferred_items]
    return sorted_items

def calc_date_range(selection):
    """
    Calculates the date range based on the selected range option.
    Returns the start and end dates for the specified range.
    """
    range_map = {
        'Last 7 Days': 7,
        'Last 30 Days': 30,
        'Last 3 Months': 90,
        'Last 6 Months': 180,
        'Last 12 Months': 365,
        'Last 16 Months': 480
    }
    today = datetime.date.today()
    return today - datetime.timedelta(days=range_map.get(selection, 0)), today


def show_error(e):
    """
    Displays an error message in the Streamlit app.
    Formats and shows the provided error 'e'.
    """
    st.error(f"An error occurred: {e}")


def property_change():
    """
    Updates the 'selected_property' in the Streamlit session state.
    Triggered on change of the property selection.
    """
    st.session_state.selected_property = st.session_state['selected_property_selector']


# -------------
# File & Download Operations
# -------------

def show_dataframe(report):
    """
    Shows a preview of the first 100 rows of the report DataFrame in an expandable section.
    """
    with st.expander("Preview the First 100 Rows"):
        st.dataframe(report.head(DF_PREVIEW_ROWS))


def download_csv_link(report):
    """
    Generates and displays a download link for the report DataFrame in CSV format.
    """
    def to_csv(df):
        return df.to_csv(index=False, encoding='utf-8-sig')

    csv = to_csv(report)
    b64_csv = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64_csv}" download="search_console_data.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)


# -------------
# Streamlit UI Components
# -------------

def show_google_sign_in(auth_url):
    """
    Displays the Google sign-in button and authentication URL in the Streamlit sidebar.
    """
    with st.sidebar:
        if st.button("Sign in with Google"):
            # Open the authentication URL
            st.write('Please click the link below to sign in:')
            st.markdown(f'[Google Sign-In]({auth_url})', unsafe_allow_html=True)


def show_property_selector(properties, account):
    """
    Displays a dropdown selector for Google Search Console properties.
    Returns the selected property's webproperty object.
    """
    selected_property = st.selectbox(
        "Select a Search Console Property:",
        properties,
        index=properties.index(
            st.session_state.selected_property) if st.session_state.selected_property in properties else 0,
        key='selected_property_selector',
        on_change=property_change
    )
    return account[selected_property]



def show_date_range_selector():
    """
    Displays a dropdown selector for choosing the date range.
    Returns the selected date range option.
    """
    return st.selectbox(
        "Select Date Range:",
        DATE_RANGE_OPTIONS,
        index=DATE_RANGE_OPTIONS.index(st.session_state.selected_date_range),
        key='date_range_selector'
    )

def show_min_clicks_input():
    """
    Displays a number input for specifying the minimum number of clicks.
    Updates the session state with the selected value.
    """
    min_clicks = st.number_input("Minimum Number of Clicks:", min_value=0, value=st.session_state.selected_min_clicks)
    st.session_state.selected_min_clicks = min_clicks
    return min_clicks

def show_directory_input():
    """
    Displays a text input for specifying an optional directory for filtering URLs.
    Returns the entered directory value.
    """
    directory = st.text_input("Directory Filter (Optional)", "", help="Enter a directory (e.g., '/de/') to filter URLs. You could even filter for subdomains(e.g., 'good.' to filter for 'good.example.com'). Leave empty to include all URLs. Remember to enter the correct symbols before and after")
    return directory

def show_fetch_data_button(webproperty, search_type, start_date, end_date, selected_dimensions, min_clicks, directory):
    """
    Displays a button to fetch data based on selected parameters.
    Shows the report DataFrame and download link upon successful data fetching.
    """
    if st.button("Fetch Data"):
        report = fetch_data_loading(webproperty, search_type, start_date, end_date, selected_dimensions, min_clicks, directory)
        if report is not None:
            st.session_state.fetched_data = report  # Store in session state

# ---------------------------
# Entity Extraction Functions
# ---------------------------
def handle_api_keys():
        model = GROQ_MODELS[0]
        if model in GROQ_MODELS:
            llm_client  = Groq(api_key=st.secrets["groq"]["api_key"])
        elif model in ANTHROPIC_MODELS:
            llm_client  = Anthropic(api_key=st.secrets["anthropic"]["api_key"])
        elif model in OPENAI_MODELS:
            llm_client  = OpenAI(api_key=st.secrets["openai"]["api_key"])
        return llm_client, model


def extract_entities_from_queries(llm_client, model, fetched_data, country, language):
    prompt = f"""
    You are a specialized assistant trained to extract the main entity or topic from a search query. Your task is to:
    - Examine the provided search query
    - Determine the main entity or topic that the query is referring to
    - If you are unsure about the main entity or topic, return None
    - Respond with the extracted entity or topic as a string, without any other text or explanations

    Given the search query from {country} in {language}, please provide the main entity or topic as a single term or phrase. Use ontologies, word embeddings, and similarity measures to identify the most relevant entity or topic. The goal is to contribute to building a comprehensive and insightful semantic map, so aim for high-quality and relevant entities.

    Your response must be in {language}.
    """

    def extract_entity(query):
        if model in ANTHROPIC_MODELS:
            try:
                response = llm_client.messages.create(
                    model=model,
                    system=prompt.format(query=query),
                    max_tokens=MAX_TOKEN,
                    temperature=TEMPERATURE,
                    messages=[
                        {"role": "user", "content": query}
                    ]
                )
                result = response.content[0].text
                return result
            except Exception as e:
                print(f"Error: {e}. Retrying in 7 seconds...")
                time.sleep(7)
        else:
            try:
                response = llm_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompt.format(query=query)},
                        {"role": "user", "content": query}],
                    max_tokens=MAX_TOKEN,
                    temperature=TEMPERATURE,
                )
                result = response.choices[0].message.content
                return result  
            except Exception as e:
                print(f"Error: {e}. Retrying in 7 seconds...")
                time.sleep(7)
    
    entities = []
    progress_bar = st.progress(0)
    for i in stqdm(range(len(fetched_data)), desc="Extracting entities"):
        query = fetched_data['query'].iloc[i]
        entity = extract_entity(query)
        entities.append(entity)
        progress_value = (i + 1) / len(fetched_data)
        progress_bar.progress(progress_value)
    
    fetched_data['entity'] = entities
    return fetched_data


class RelationshipGenerator:
    """
    A class for generating relationships between entities.
    """
    def __init__(self, llm_client, model):
        """
        Initializes the RelationshipGenerator instance.
        Args:
            llm_client: The LLM client instance to use for generating relationships.
            model: The LLM model to use for generating relationships.
        """
        self.llm_client = llm_client
        self.model = model

    def generate_batch_relationships(self, batch_entities, other_entities, existing_relationships, country, language):
        """
        Generates relationships between a batch of entities and other entities.
        Args:
            batch_entities (Dict[str, str]): A dictionary of entities in the current batch, where keys are entity IDs and values are entity labels.
            other_entities (Dict[str, str]): A dictionary of other entities, where keys are entity IDs and values are entity labels.
            existing_relationships (Set[tuple]): A set of existing relationships, where each tuple represents a relationship (source_id, target_id, edge_label).
            country (str): The country for which the relationships are being generated.
            language (str): The language in which the relationships should be generated.
        Returns:
            Set[tuple]: A set of new relationships, where each tuple represents a relationship (source_id, target_id, edge_label).
        """
        prompt = f"""
        Given the following entities from {country} in {language}:
        {{batch_entities}}
        Consider the other entities:
        {{other_entities}}
        and the existing relationships:
        {{existing_relationships}}
        Your task is to identify relevant relationships between the given entities and the other entities.
        Use domain knowledge to prioritize important connections and provide meaningful edge labels. You must give each entity no less than 2 relationships and no more than 5 relationships for any individual entity. You must return all requested entity relationships.
        Example output:
        source_id1,target_id1,edge_label1
        source_id2,target_id2,edge_label2
        source_id3,target_id3,edge_label3
        Please provide the output as a list of relationships and their labels, in the format 'source_id,target_id,edge_label', without any other text or explanations.
        Focus on identifying the most significant and impactful relationships.
        Your response must be in {language}.
        """

        def generate_relationships(query):
            if self.model in ANTHROPIC_MODELS:
                try:
                    response = self.llm_client.messages.create(
                        model=self.model,
                        system=prompt.format(batch_entities=query[0], other_entities=query[1], existing_relationships=query[2]),
                        max_tokens=MAX_TOKEN,
                        temperature=TEMPERATURE,
                        messages=[
                            {"role": "user", "content": ""}
                        ]
                    )
                    result = response.content[0].text
                    return result
                except Exception as e:
                    print(f"Error: {e}. Retrying in 7 seconds...")
                    time.sleep(7)
            else:
                try:
                    response = self.llm_client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": prompt.format(batch_entities=query[0], other_entities=query[1], existing_relationships=query[2])},
                            {"role": "user", "content": ""}],
                        max_tokens=MAX_TOKEN,
                        temperature=TEMPERATURE,
                    )
                    result = response.choices[0].message.content
                    return result
                except Exception as e:
                    print(f"Error: {e}. Retrying in 7 seconds...")
                    time.sleep(7)

        batch_entity_ids = list(batch_entities.keys())
        existing_batch_relationships = [f"{rel[0]},{rel[1]},{rel[2]}" for rel in existing_relationships if rel[0] in batch_entity_ids]
        query = (
            ", ".join([f"{id}: {entity}" for id, entity in batch_entities.items()]),
            ", ".join([f"{id}: {entity}" for id, entity in other_entities.items()]),
            ", ".join(existing_batch_relationships)
        )
        new_relationships_response = generate_relationships(query)
        new_relationships = set()
        for rel in new_relationships_response.split("\n"):
            rel = rel.strip()
            if "," in rel:
                parts = rel.split(",")
                if len(parts) >= 2:
                    source_id, target_id = parts[:2]
                    source_id = source_id.strip()
                    target_id = target_id.strip()
                    edge_label = parts[2].strip() if len(parts) > 2 else ""
                    if source_id in batch_entity_ids and target_id in other_entities and (source_id, target_id, edge_label) not in existing_relationships:
                        new_relationships.add((source_id, target_id, edge_label))
        return new_relationships

    def generate_relationships(self, entities, existing_relationships, batch_size, num_parallel_runs, country, language):
        """
        Generates relationships between entities in parallel.
        Args:
            entities (Dict[str, str]): A dictionary of entities, where keys are entity IDs and values are entity labels.
            existing_relationships (Set[tuple]): A set of existing relationships, where each tuple represents a relationship (source_id, target_id, edge_label).
            batch_size (int): The size of the batches for parallel processing.
            num_parallel_runs (int): The number of parallel runs to perform.
            country (str): The country for which the relationships are being generated.
            language (str): The language in which the relationships should be generated.
        Returns:
            Set[tuple]: A set of new relationships, where each tuple represents a relationship (source_id, target_id, edge_label).
        """
        new_relationships = set()
        entity_ids = list(entities.keys())
        batches = [entity_ids[i:i+batch_size] for i in range(0, len(entity_ids), batch_size)]
        progress_bar = st.progress(0)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel_runs) as executor:
            futures = []
            for batch_entity_ids in batches:
                batch_entities = {id: entities[id] for id in batch_entity_ids}
                other_entities = {id: entities[id] for id in entities if id not in batch_entity_ids}
                future = executor.submit(self.generate_batch_relationships, batch_entities, other_entities, existing_relationships, country, language)
                futures.append(future)
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                new_relationships.update(future.result())
                progress_value = (i + 1) / len(futures)
                progress_bar.progress(progress_value)
        return new_relationships


# -------------
# Main Streamlit App Function
# -------------

# Main Streamlit App Function
def main():
    """
    The main function for the Streamlit application.
    Handles the app setup, authentication, UI components, and data fetching logic.
    """
    setup_streamlit()
    client_config = load_config()
    st.session_state.auth_flow, st.session_state.auth_url = google_auth(client_config)

    auth_code = st.query_params.get("code", None)

    if auth_code and not st.session_state.get('credentials'):
        st.session_state.auth_flow.fetch_token(code=auth_code)
        st.session_state.credentials = st.session_state.auth_flow.credentials

    if not st.session_state.get('credentials'):
        show_google_sign_in(st.session_state.auth_url)
    else:
        init_session_state()
        account = auth_search_console(client_config, st.session_state.credentials)
        properties = list_gsc_properties(st.session_state.credentials)

        if properties:
            webproperty = show_property_selector(properties, account)
            search_type = SEARCH_TYPE
            date_range_selection = show_date_range_selector()
            start_date, end_date = calc_date_range(date_range_selection)
            selected_dimensions = DIMENSIONS
            min_clicks = show_min_clicks_input()
            sorted_countries = custom_sort(COUNTRIES, preferred_countries)
            sorted_languages = custom_sort(LANGUAGES, preferred_languages)
            llm_client, model = handle_api_keys()
            country = st.selectbox("Country", sorted_countries)
            language = st.selectbox("Language", sorted_languages)
            directory = show_directory_input()
            show_fetch_data_button(webproperty, search_type, start_date, end_date, selected_dimensions, min_clicks, directory)
            if 'fetched_data' in st.session_state and st.session_state.fetched_data is not None:
                fetched_data = st.session_state.fetched_data
                show_dataframe(fetched_data)
                #fetched_data = extract_entities_from_queries(llm_client, model, fetched_data, country, language)
                #st.session_state.fetched_data_with_entities = fetched_data
                # Create an instance of RelationshipGenerator
                #relationship_generator = RelationshipGenerator(llm_client, model)
            
                # Generate relationships
                #new_relationships = relationship_generator.generate_relationships(entities, existing_relationships, batch_size, num_parallel_runs, country, language)
                    


if __name__ == "__main__":
    main()

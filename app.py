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
nltk.download('stopwords')

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

# Define models
ANTHROPIC_MODELS = ['claude-3-opus-20240229', 'claude-3-sonnet-20240229','claude-3-haiku-20240307']
GROQ_MODELS = ['mixtral-8x7b-32768', 'llama3-70b-8192']
OPENAI_MODELS = ['gpt-4-turbo', 'gpt-3.5-turbo']
MODELS = GROQ_MODELS + ANTHROPIC_MODELS + OPENAI_MODELS
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
        page_title="Semantic Sitemap Generator with Topical Authority",
        page_icon=":crown:",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.linkedin.com/in/kirchhoff-kevin/',
            'About': "This is an app for generating semantic sitemaps and analyzing topical authority! Adapted from Lee Foot's GSC-connector check out his apps: https://leefoot.co.uk"
        }
    )
    st.image("https://www.claneo.com/wp-content/uploads/Element-4.svg", width=600, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    st.caption(":point_right: Join Claneo and support exciting clients as part of the Consulting team") 
    st.caption(':bulb: Make sure to mention that *Kevin* brought this job posting to your attention')
    st.link_button("Learn More", "https://www.claneo.com/en/career/#:~:text=Consulting")
    st.title("Semantic Sitemap Generator with Topical Authority Analysis")
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


def fetch_gsc_data(webproperty, search_type, start_date, end_date, dimensions, min_clicks, device_type=None):
    """
    Fetches Google Search Console data for a specified property, date range, dimensions, and device type.
    Handles errors and returns the data as a DataFrame.
    """
    query = webproperty.query.range(start_date, end_date).search_type(search_type).dimension(*dimensions)

    try:
        df = query.limit(MAX_ROWS).get().to_dataframe()
        if 'clicks' in df.columns and 'position' in df.columns:
            df = df[df['clicks'] >= min_clicks]
            
        else:
            show_error("Columns 'clicks' or 'position' not in DataFrame")
        return df
    except Exception as e:
        show_error(e)
        return pd.DataFrame()


def fetch_data_loading(webproperty, search_type, start_date, end_date, dimensions, min_clicks, device_type=None):
    """
    Fetches Google Search Console data with a loading indicator. Utilises 'fetch_gsc_data' for data retrieval.
    Returns the fetched data as a DataFrame.
    """
    with st.spinner('Fetching data...'):
        return fetch_gsc_data(webproperty, search_type, start_date, end_date, dimensions, device_type, min_clicks)


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

def extract_main_queries(df, min_clicks):
    """
    Extracts the main queries for each URL where the main keyword is in the top 5 positions and generates a minimum amount of traffic.
    Returns a DataFrame with the extracted main queries and their corresponding URLs.
    """
    main_queries_df = df[(df['position'] <= 5) & (df['clicks'] >= min_clicks)][['page', 'query']]
    main_queries_df = main_queries_df.groupby('page').agg({'query': lambda x: x.iloc[0]}).reset_index()
    main_queries_df.columns = ['URL', 'Main Query']
    return main_queries_df

def show_fetch_data_button(webproperty, search_type, start_date, end_date, selected_dimensions, min_clicks):
    """
    Displays a button to fetch data based on selected parameters.
    Shows the report DataFrame and download link upon successful data fetching.
    """
    if st.button("Fetch Data"):
        report = fetch_data_loading(webproperty, search_type, start_date, end_date, selected_dimensions, min_clicks)
        if report is not None:
            st.session_state.fetched_data = report  # Store in session state
            #main_queries_df = extract_main_queries(report, min_clicks)
            #st.session_state.main_queries_df = main_queries_df  # Store in session state


# ---------------------------
# Entity Extraction Functions
# ---------------------------
def handle_api_keys():
        model = st.selectbox("Choose a model:", MODELS, help=f"""
        Here's a brief overview of the models available for generating content:
        
        - **{GROQ_MODELS}**: These models are free to use and offer fast response times, making them an excellent choice for users looking for quick results. However, they may not always provide the highest quality of text. Among the GROQ models, the first model in this list: **{GROQ_MODELS[0]}** is generally considered the best due to its balance of speed and quality.
        
        - **{ANTHROPIC_MODELS}**: The models from Anthropic are known for their superior text quality. However, they require an API key, which can be obtained from [Anthropic's platform](https://console.anthropic.com/settings/keys). Among the Anthropic models, the first model in this list: **{ANTHROPIC_MODELS[0]}**, is considered the best, offering the highest quality text, but is the most costly.
        
        - **{OPENAI_MODELS}**: These are the most well-known models in the industry. You can obtain an API key from [OpenAI's platform](https://platform.openai.com/api-keys). Among the OpenAI models, the first model in this list: **{OPENAI_MODELS[0]}**, is considered the best, offering the highest quality text, but is the most costly.
        
        **It's important to note that the quality and cost-effectiveness of models can vary greatly, so choose the model wisely and test before creating loads of meta data. Always consider your specific needs and budget when selecting a model.**
        
        For the most current information on which model is performing best overall, you can visit the [Chatbot Arena Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) on Hugging Face. This leaderboard provides insights into the performance of various models in real-world scenarios, helping you make an informed decision.
        """)
        if model in GROQ_MODELS:
            llm_client  = Groq(api_key=st.secrets["groq"]["api_key"])
        elif model in ANTHROPIC_MODELS:
            llm_client  = Anthropic(api_key=st.secrets["anthropic"]["api_key"])
        elif model in OPENAI_MODELS:
            llm_client  = OpenAI(api_key=st.secrets["openai"]["api_key"])
        return llm_client, model

def extract_entities_from_queries(llm_client, model, main_queries_df, country, language):
    prompt = f"""
    You are a specialized assistant trained to extract the main entity or topic from a search query. Your task is to:
    - Examine the provided search query
    - Determine the main entity or topic that the query is referring to
    - If you are unsure about the main entity or topic, return None
    - Respond with the extracted entity or topic as a string

    Your response must be in {language}, note that the input is from {country} in {language}
    """

    def extract_entity(query):
        if model in ANTHROPIC_MODELS:
            try:
                response = llm_client.messages.create(
                    model=model,
                    system=prompt,
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
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": query}],
                    max_tokens=MAX_TOKEN,
                    temperature=TEMPERATURE,
                )
                result = response.choices[0].message.content
                return result  
            except Exception as e:
                print(f"Error: {e}. Retrying in 7 seconds...")
                time.sleep(7)
        print(result)
    main_queries_df['Entity'] = main_queries_df['Main Query'].apply(extract_entity)
    return main_queries_df


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
            show_fetch_data_button(webproperty, search_type, start_date, end_date, selected_dimensions, min_clicks)
            show_dataframe(st.session_state.fetched_data)
            #if 'main_queries_df' in st.session_state and st.session_state.main_queries_df is not None:
             #   main_queries_df = st.session_state.main_queries_df
              #  st.write('Before extraction')
               # show_dataframe(main_queries_df)
                #main_queries_df = extract_entities_from_queries(llm_client, model, main_queries_df, country, language)
                #show_dataframe(main_queries_df)
                
                    


if __name__ == "__main__":
    main()

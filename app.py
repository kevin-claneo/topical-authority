# Standard library imports
import datetime
import base64
import concurrent.futures
import time

# Related third-party imports
import streamlit as st
from streamlit_elements import Elements
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import pandas as pd
import searchconsole#from stqdm import stqdm
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Local imports
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_anthropic import ChatAnthropic
import anthropic
import networkx as nx
import community as community_louvain
from typing import Dict, Set
import Levenshtein
from rake_nltk import Rake
import spacy
from spacy_entity_linker import EntityLinker

# Constants
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
Opus = "claude-3-opus-20240229"
Sonnet = "claude-3-sonnet-20240229"
Haiku = "claude-3-haiku-20240307"

# -------------
# Streamlit App Configuration
# -------------

# Streamlit App Configuration
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

def extract_topics(queries, num_topics, temperature):
    topic_generator = EntityGenerator(llm)
    topics = topic_generator.generate_entities(" ".join(queries), {}, num_topics, temperature)
    return topics

def extract_main_queries(df, min_clicks):
    """
    Extracts the main queries for each URL where the main keyword is in the top 5 positions and generates a minimum amount of traffic.
    Returns a DataFrame with the extracted main queries and their corresponding URLs.
    """
    main_queries_df = df[(df['position'] <= 5) & (df['clicks'] >= min_clicks)][['page', 'query']]
    main_queries_df = main_queries_df.groupby('page').agg({'query': lambda x: x.iloc[0]}).reset_index()
    main_queries_df.columns = ['URL', 'Main Query']
    return main_queries_df

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
    with st.sidebar.spinner('Fetching data...'):
        return fetch_gsc_data(webproperty, search_type, start_date, end_date, dimensions, device_type, min_clicks)


# -------------
# Utility Functions
# -------------


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
    return st.sidebar.selectbox(
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
    min_clicks = st.sidebar.number_input("Minimum Number of Clicks:", min_value=0, value=st.session_state.selected_min_clicks)
    st.session_state.selected_min_clicks = min_clicks
    return min_clicks


def show_fetch_data_button(webproperty, search_type, start_date, end_date, selected_dimensions, min_clicks):
    """
    Displays a button to fetch data based on selected parameters.
    Shows the report DataFrame and download link upon successful data fetching.
    """
    if st.sidebar.button("Fetch Data"):
        report = fetch_data_loading(webproperty, search_type, start_date, end_date, selected_dimensions, min_clicks)
        if report is not None:
            st.session_state.fetched_data = report  # Store in session state
            main_queries_df = extract_main_queries(report, min_clicks)
            st.session_state.main_queries_df = main_queries_df  # Store in session state


# ---------------------------
# Entity Extraction Functions
# ---------------------------

def extract_topics(queries, num_topics, temperature):
    topic_generator = EntityGenerator(llm)
    topics = topic_generator.generate_entities(" ".join(queries), {}, num_topics, temperature)
    return topics

def extract_main_queries(df, min_clicks):
    """
    Extracts the main queries for each URL where the main keyword is in the top 5 positions and generates a minimum amount of traffic.
    Returns a DataFrame with the extracted main queries and their corresponding URLs.
    """
    main_queries_df = df[(df['position'] <= 5) & (df['clicks'] >= min_clicks)][['page', 'query']]
    main_queries_df = main_queries_df.groupby('page').agg({'query': lambda x: x.iloc[0]}).reset_index()
    main_queries_df.columns = ['URL', 'Main Query']
    return main_queries_df

class LLMCaller:
    @staticmethod
    def make_llm_call(args):
        """
        Makes a call to the Anthropic LLM API with the provided arguments.
        Args:
            args (dict): A dictionary containing the following keys:
                - api_key (str): The Anthropic API key for authentication.
                - system_prompt (str): The system prompt for the LLM.
                - prompt (str): The user prompt for the LLM.
                - model_name (str): The name of the Claude model to use.
                - max_tokens (int): The maximum number of tokens for the LLM response.
                - temperature (float): The temperature value for the LLM response.
        Returns:
            str: The response from the LLM, or None if an exception occurred.
        """
        try:
            response = anthropic.Anthropic(api_key=args["api_key"]).messages.create(
                system=args["system_prompt"],
                messages=[{"role": "user", "content": args["prompt"]}],
                model=args["model_name"],
                max_tokens=args["max_tokens"],
                temperature=args["temperature"],
                stop_sequences=[],
            )
            return response.content[0].text
        except Exception as e:
            print(f"Error making LLM call: {e}")
            return None


class EntityGenerator:
    """
    A class for generating new entities related to a given topic.
    """
    def __init__(self, llm):
        """
        Initializes the EntityGenerator instance.
        Args:
            llm (LLM): The LLM instance to use for generating entities.
        """
        self.llm = llm
        self.entity_id_counter = 0

    def generate_entities(self, topic: str, existing_entities: Dict[str, str], num_new_entities: int, temperature: float) -> Dict[str, str]:
        """
        Generates new entities related to the given topic.
        Args:
            topic (str): The topic for which to generate entities.
            existing_entities (Dict[str, str]): A dictionary of existing entities, where keys are entity IDs and values are entity labels.
            num_new_entities (int): The number of new entities to generate.
            temperature (float): The temperature value for the LLM response.
        Returns:
            Dict[str, str]: A dictionary of new entities, where keys are entity IDs and values are entity labels.
        """
        prompt = PromptTemplate(
            input_variables=["topic", "existing_entities", "num_new_entities"],
            template="""Given the topic '{topic}' and the existing entities:\n\n{existing_entities}\n\n
            Your task is to suggest {num_new_entities} new entities that are semantically related to the topic and existing entities, but not already present in the existing entities.
            Use ontologies, word embeddings, and similarity measures to expand the entities while narrowing the focus based on the existing entities. Employ a simulated Monte Carlo Tree search as your mental model for coming up with this list. The goal is complete comprehensive semantic depth and breadth for the topic.
            Example output:
            machine learning, deep learning, neural networks, computer vision, natural language processing
            Please provide the output as a comma-separated list of new entities, without any other text or explanations. Your suggestions will contribute to building a comprehensive and insightful semantic map, so aim for high-quality and relevant entities.""",
        )
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        new_entities_response = llm_chain.run(
            topic=topic,
            existing_entities=", ".join([entity for entity in existing_entities.values()]),
            num_new_entities=num_new_entities,
        )
        new_entities = {}
        for entity in new_entities_response.split(","):
            entity = entity.strip()
            if entity and entity not in existing_entities.values():
                entity_id = f"e{self.entity_id_counter}"
                new_entities[entity_id] = entity
                self.entity_id_counter += 1
        return new_entities


class RelationshipGenerator:
    """
    A class for generating relationships between entities.
    """
    def __init__(self, llm):
        """
        Initializes the RelationshipGenerator instance.
        Args:
            llm (LLM): The LLM instance to use for generating relationships.
        """
        self.llm = llm

    def generate_batch_relationships(self, topic: str, batch_entities: Dict[str, str], other_entities: Dict[str, str], existing_relationships: Set[tuple]) -> Set[tuple]:
        """
        Generates relationships between a batch of entities and other entities.
        Args:
            topic (str): The topic for which to generate relationships.
            batch_entities (Dict[str, str]): A dictionary of entities in the current batch, where keys are entity IDs and values are entity labels.
            other_entities (Dict[str, str]): A dictionary of other entities, where keys are entity IDs and values are entity labels.
            existing_relationships (Set[tuple]): A set of existing relationships, where each tuple represents a relationship (source_id, target_id, edge_label).
        Returns:
            Set[tuple]: A set of new relationships, where each tuple represents a relationship (source_id, target_id, edge_label).
        """
        prompt = PromptTemplate(
            input_variables=["topic", "batch_entities", "other_entities", "existing_relationships"],
            template="""Given the topic '{topic}' and the following entities:
            {batch_entities}
            Consider the other entities:
            {other_entities}
            and the existing relationships:
            {existing_relationships}
            Your task is to identify relevant relationships between the given entities and the other entities in the context of the topic.
            Use domain knowledge to prioritize important connections and provide meaningful edge labels. You must give each entity no less than 2 relationships and no more than 5 relationships for any individual entity. You must return all requested entity relationships.
            Example output:
            source_id1,target_id1,edge_label1
            source_id2,target_id2,edge_label2
            source_id3,target_id3,edge_label3
            Please provide the output as a list of relationships and their labels, in the format 'source_id,target_id,edge_label', without any other text or explanations.
            Focus on identifying the most significant and impactful relationships.""",
        )
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        batch_entity_ids = list(batch_entities.keys())
        existing_batch_relationships = [f"{rel[0]},{rel[1]},{rel[2]}" for rel in existing_relationships if rel[0] in batch_entity_ids]
        new_relationships_response = llm_chain.run(
            topic=topic,
            batch_entities=", ".join([f"{id}: {entity}" for id, entity in batch_entities.items()]),
            other_entities=", ".join([f"{id}: {entity}" for id, entity in other_entities.items()]),
            existing_relationships=", ".join(existing_batch_relationships),
        )
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
    
    def generate_relationships(self, topic: str, entities: Dict[str, str], existing_relationships: Set[tuple], batch_size: int, num_parallel_runs: int) -> Set[tuple]:
        """
        Generates relationships between entities in parallel.
        Args:
            topic (str): The topic for which to generate relationships.
            entities (Dict[str, str]): A dictionary of entities, where keys are entity IDs and values are entity labels.
            existing_relationships (Set[tuple]): A set of existing relationships, where each tuple represents a relationship (source_id, target_id, edge_label).
            batch_size (int): The size of the batches for parallel processing.
            num_parallel_runs (int): The number of parallel runs to perform.
        Returns:
            Set[tuple]: A set of new relationships, where each tuple represents a relationship (source_id, target_id, edge_label).
        """
        new_relationships = set()
        entity_ids = list(entities.keys())
        batches = [entity_ids[i:i+batch_size] for i in range(0, len(entity_ids), batch_size)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel_runs) as executor:
            futures = []
            for batch_entity_ids in batches:
                batch_entities = {id: entities[id] for id in batch_entity_ids}
                other_entities = {id: entities[id] for id in entities if id not in batch_entity_ids}
                for _ in range(num_parallel_runs):
                    future = executor.submit(self.generate_batch_relationships, topic, batch_entities, other_entities, existing_relationships)
                    futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                new_relationships.update(future.result())
        return new_relationships


class SemanticMapGenerator:
    """
    A class for generating a semantic map based on entities and relationships.
    """
    def __init__(self, entity_generator: EntityGenerator, relationship_generator: RelationshipGenerator):
        """
        Initializes the SemanticMapGenerator instance.
        Args:
            entity_generator (EntityGenerator): The EntityGenerator instance to use for generating entities.
            relationship_generator (RelationshipGenerator): The RelationshipGenerator instance to use for generating relationships.
        """
        self.entity_generator = entity_generator
        self.relationship_generator = relationship_generator
        self.entities = {}
        self.relationships = set()

    def generate_semantic_map(self, topic: str, num_iterations: int, num_parallel_runs: int, num_entities_per_run: int, temperature: float, relationship_batch_size: int) -> Dict[str, Set]:
        """
        Generates a semantic map for the given topic.
        Args:
            topic (str): The topic for which to generate the semantic map.
            num_iterations (int): The number of iterations to perform for generating entities and relationships.
            num_parallel_runs (int): The number of parallel runs to perform for entity and relationship generation.
            num_entities_per_run (int): The number of new entities to generate in each run.
            temperature (float): The temperature value for the LLM response.
            relationship_batch_size (int): The size of the batches for parallel relationship generation.
        Returns:
            Dict[str, Set]: A dictionary containing the generated entities and relationships, where the keys are 'entities' and 'relationships', and the values are sets of entities and relationships, respectively.
        """
        entities_count = 0
        relationships_count = 0
        entities_placeholder = st.empty()
        relationships_placeholder = st.empty()
        for iteration in stqdm(range(num_iterations), desc="Generating Semantic Map"):
            # Parallel entity generation
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel_runs) as executor:
                futures = []
                for _ in range(num_parallel_runs):
                    future = executor.submit(self.entity_generator.generate_entities, topic, self.entities, num_entities_per_run, temperature)
                    futures.append(future)
                progress = stqdm(total=num_parallel_runs, desc="Generating Entities", leave=False)
                progress.update(1)
                new_entities = {}
                for future in concurrent.futures.as_completed(futures):
                    new_entities.update(future.result())
                    progress.update(-1)
                    progress.update(1)
                    time.sleep(0.1)  # Simulate progress
                progress.close()
            # Deduplicate entities
            self.entities.update(new_entities)
            entities_count += len(new_entities)
            # Parallel relationship generation
            new_relationships = self.relationship_generator.generate_relationships(topic, self.entities, self.relationships, relationship_batch_size, num_parallel_runs)
            self.relationships.update(new_relationships)
            relationships_count += len(new_relationships)
            # Simulate intermediate progress for relationship generation
            for _ in range(num_parallel_runs):
                progress = (iteration * num_parallel_runs + _ + 1) / (num_iterations * num_parallel_runs)
                progress_bar.progress(progress)
                time.sleep(0.1)
            # Update metrics
            entities_placeholder.metric("Total Entities", entities_count)
            relationships_placeholder.metric("Total Relationships", relationships_count)
        return {"entities": self.entities, "relationships": self.relationships}


def save_semantic_map_to_csv(semantic_map: Dict[str, Set], topic: str):
    """
    Saves the generated semantic map to CSV files.
    Args:
        semantic_map (Dict[str, Set]): A dictionary containing the generated entities and relationships.
        topic (str): The topic for which the semantic map was generated.
    """
    entities_file = f"{topic}_entities.csv"
    with open(entities_file, "w") as f:
        f.write("Id,Label\n")
        progress = stqdm(semantic_map["entities"].items(), desc="Saving Entities to CSV", total=len(semantic_map["entities"]))
        for id, entity in progress:
            f.write(f"{id},{entity}\n")
            time.sleep(0.01)  # Simulate progress
    relationships_file = f"{topic}_relationships.csv"
    with open(relationships_file, "w") as f:
        f.write("Source,Target,Type\n")
        progress = stqdm(semantic_map["relationships"], desc="Saving Relationships to CSV", total=len(semantic_map["relationships"]))
        for relationship in progress:
            f.write(f"{relationship[0]},{relationship[1]},{relationship[2]}\n")
            time.sleep(0.01)  # Simulate progress

def merge_similar_nodes(G, similarity_threshold=0.8):
    """
    Merges similar nodes in the graph based on their label similarity.
    Args:
        G (NetworkX graph): The graph to merge similar nodes in.
        similarity_threshold (float, optional): The threshold for label similarity. Defaults to 0.8.
    Returns:
        NetworkX graph: The graph with similar nodes merged.
    """
    merged_nodes = set()
    for node1 in G.nodes():
        if node1 not in merged_nodes:
            for node2 in G.nodes():
                if node1 != node2 and node2 not in merged_nodes:
                    label1 = G.nodes[node1]['label']
                    label2 = G.nodes[node2]['label']
                    similarity = Levenshtein.ratio(label1, label2)
                    if similarity >= similarity_threshold:
                        # Merge nodes
                        G = nx.contracted_nodes(G, node1, node2, self_loops=False)
                        merged_nodes.add(node2)
                        break
    return G


def visualize_semantic_map(semantic_map: Dict[str, Set]):
    """
    Visualizes the semantic map using NetworkX and Graphviz via Streamlit's graphviz_chart.
    Args:
        semantic_map (Dict[str, Set]): A dictionary containing the generated entities and relationships.
    """
    G = nx.Graph()
    for entity_id, entity_label in semantic_map["entities"].items():
        G.add_node(entity_id, label=entity_label)
    for source_id, target_id, edge_label in semantic_map["relationships"]:
        G.add_edge(source_id, target_id, label=edge_label)
    
    # Merge similar nodes
    G = merge_similar_nodes(G)
    
    # Perform community detection using Louvain algorithm
    partition = community_louvain.best_partition(G)
    
    # Set node attributes for community and label
    for node in G.nodes():
        G.nodes[node]['community'] = str(partition[node])
        G.nodes[node]['label'] = G.nodes[node]['label']
    
    # Set edge attributes for label
    for edge in G.edges():
        G.edges[edge]['label'] = G.edges[edge]['label']
    
    # Convert NetworkX graph to PyDot, then to DOT format
    pydot_graph = to_pydot(G)
    
    # Set node attributes for style based on community
    for node in pydot_graph.get_nodes():
        node.set_style('filled')
        node.set_fillcolor(f'/{node.get_attributes()["community"]}')
    
    # Set edge attributes for label
    for edge in pydot_graph.get_edges():
        edge.set_label(edge.get_attributes()['label'])
    
    dot_string = pydot_graph.to_string()
    
    # Use Streamlit's graphviz_chart to visualize the graph
    st.graphviz_chart(dot_string)

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
            show_fetch_data_button(webproperty, search_type, start_date, end_date, selected_dimensions, min_clicks)
            if 'main_queries_df' in st.session_state and st.session_state.main_queries_df is not None:
                main_queries_df = st.session_state.main_queries_df
                main_queries = main_queries_df['Main Query'].tolist()
                
                # Generate topics from the main queries
                num_topics = 10  # Specify the desired number of topics
                temperature = 0.7  # Specify the temperature for topic generation
                topics = extract_topics(main_queries, num_topics, temperature)
                
                # Generate relationships between the topics
                relationship_generator = RelationshipGenerator(llm)
                relationships = relationship_generator.generate_relationships(" ".join(main_queries), topics, set(), 5, 2)
                
                # Create a semantic map based on the topics and relationships
                semantic_map_generator = SemanticMapGenerator(None, relationship_generator)
                semantic_map = {"entities": topics, "relationships": relationships}
                
                # Display the semantic map interactively
                visualize_semantic_map(semantic_map)


if __name__ == "__main__":
    main()

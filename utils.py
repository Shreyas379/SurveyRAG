import base64
from typing import List, Dict

from haystack import Document, Pipeline
from haystack import component
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.cohere.document_embedder import CohereDocumentEmbedder
from haystack_integrations.components.embedders.cohere.text_embedder import CohereTextEmbedder
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.utils import Secret
import pandas as pd

# ***************************************************Data preprocessing & data parsing functions***************************************************
def rename_null_values(process_list):
    """
    Replaces null (NaN) values in a list with the previous valid value, 
    appending '_percentage' to it.
    
    Parameters:
    process_list (list): A list of demographic values that may contain NaN values.

    Returns:
    list: A list with NaN values replaced by the previous valid value with '_percentage' appended.
    """
    # Initialize an empty list to store the updated values
    updated_list = []

    # Variable to store the last valid (non-NaN) value
    last_valid_value = None

    # Iterate through each value in the list
    for value in process_list:
        if pd.isna(value):  # If the current value is NaN
            if last_valid_value is not None:  # Ensure a previous valid value exists
                updated_list.append(f"{last_valid_value}_percentage")  # Replace NaN with last valid value + '_percentage'
        else:
            updated_list.append(value)  # Keep the original valid value
            last_valid_value = value  # Update the last valid value for future replacements

    return updated_list


def process_excelfile(df):
    """
    Processes an Excel dataframe by cleaning, renaming columns, and updating demographic data.

    This function:
    - Extracts a specific portion of the dataframe.
    - Renames certain columns for clarity.
    - Drops unnecessary columns.
    - Replaces null (NaN) values in the 'Demographics' column using the `rename_null_values` function.
    
    Parameters:
    df (DataFrame): The input dataframe representing the Excel sheet.

    Returns:
    DataFrame: The processed dataframe with cleaned column names and updated 'Demographics' values.
    """
    # Extract the relevant part of the dataframe starting from row 3 and specific columns
    df_ = df.loc[3:, 'Demographics':'55-64']

    # Rename specific columns for clarity
    df_.rename(columns={'Unnamed: 1': 'Total Sample'}, inplace=True)

    # Drop unnecessary columns that are not needed
    df_.drop([' .1', ' ', 'Unnamed: 2'], axis=1, inplace=True)

    # Get the list of demographic values
    process_list = df_['Demographics'].values.tolist()

    # Update the first value in the list (assuming it should be 'Total_percentage')
    process_list[0] = 'Total_percentage'

    # Replace NaN values in 'Demographics' column using the rename_null_values function
    df_['Demographics'] = rename_null_values(process_list)

    return df_

def generate_survey_json(df):
    """
    Generates a JSON-like dictionary structure from a DataFrame representing survey data.
    
    The function processes a DataFrame where each row contains either a survey question
    or answer choices related to a previously seen question. The resulting dictionary
    maps each question to its corresponding choices.
    
    Parameters:
    df (DataFrame): The input DataFrame where each row contains either a question or its choices.
                    The 'Demographics' column contains the questions, while the remaining columns
                    hold the potential choices for each question.
    
    Returns:
    dict: A dictionary where each key is a question and the corresponding value is a list of 
          choices formatted as "column_name: choice_value".
    """
    # Create an empty dictionary for the JSON structure
    result = {}

    # Initialize variables to track the current question
    current_question = None
    choices = []

    # Function to identify if a row is a question (this can be customized based on your data structure)
    def is_question(text):
        # Assumes a question contains the phrase 'Question ' (you can change this as needed)
        return text is not None and 'Question ' in text

    # Iterate over the DataFrame row by row
    for index, row in df.iterrows():
        first_column_value = row['Demographics']

        # Check if the current row contains a new question
        if is_question(first_column_value):
            # If there was a previous question, save it with its choices to the result dictionary
            if current_question:
                result[current_question] = choices

            # Set the new current question and reset the choices list
            current_question = first_column_value
            choices = []
        else:
            # Concatenate each column name with the corresponding value (ignoring None values)
            choice = [f"{col_name}: {row[col_name]}" for col_name in df.columns if pd.notna(row[col_name])]

            # Add the choices if there are valid (non-None) entries
            if choice:
                choices.append(choice)

    # After the loop, add the final question and its choices to the result dictionary
    if current_question:
        result[current_question] = choices

    return result

def pipeline_process_file_and_convert_to_json_records(files_data):
    """
    Processes uploaded Excel files and converts their data into JSON records.

    This function takes a list of file paths corresponding to uploaded Excel files, 
    processes each file using the `process_excelfile` function, and generates 
    JSON records using the `generate_survey_json` function.

    Args:
        files_data (list): A list containing file paths of the uploaded Excel files.

    Returns:
        tuple: A tuple containing two JSON records generated from the processed 
               Excel files.
    """
    #read excel files as pandas dataframe
    df_ds1 = pd.read_excel(files_data[0], header=1) 
    df_ds2 = pd.read_excel(files_data[1], header=1)
    
    # Process the first Excel file and store the resulting DataFrame
    df_ds1 = process_excelfile(df_ds1)
    
    # Process the second Excel file and store the resulting DataFrame
    df_ds2 = process_excelfile(df_ds2)

    # Generate JSON records from the first DataFrame
    result_file_1 = generate_survey_json(df_ds1)
    
    # Generate JSON records from the second DataFrame
    result_file_2 = generate_survey_json(df_ds2)

    # Return the generated JSON records
    return result_file_1, result_file_2
    
# ***************************************************Create vector database ( document store )***************************************************
def document_store_init(api_key: str, url: str, index_name: str = 'ServeyRAG') -> QdrantDocumentStore:
    """
    Initializes a QdrantDocumentStore for storing and retrieving documents with embeddings.
    
    Parameters:
    ----------
    api_key : str
        The API key used to authenticate with the Qdrant instance.
    url : str
        The URL of the Qdrant instance.
    index_name : str, optional
        The name of the index to use in the Qdrant document store. Defaults to 'ServeyRAG'.
    
    Returns:
    -------
    QdrantDocumentStore
        An initialized instance of QdrantDocumentStore.
    """
    
    # Initialize and configure QdrantDocumentStore with the specified parameters
    document_store = QdrantDocumentStore(
        api_key=Secret.from_token(api_key),  # Pass the API key securely using Secret handling
        url=url,                             # Set the Qdrant service URL
        recreate_index=False,                # Avoid recreating the index if it already exists
        return_embedding=True,               # Retrieve embeddings when querying the document store
        wait_result_from_api=True,           # Wait for a result from the API before proceeding
        embedding_dim=1024,                  # Dimensionality of the embeddings used for the documents
        index=index_name,                    # Name of the index to use
        similarity="cosine"                  # Set cosine similarity as the metric for comparing embeddings
    )
    
    return document_store

# ***************************************************Create indexing pipeline***************************************************
@component
class SurveyJSONToDocument:
    """
    A component that generates a list of Haystack Document objects from multiple survey JSON results.
    Each survey JSON contains questions and associated choices, which are converted into documents.
    
    Output:
    -------
    documents : List[Document]
        A list of documents generated from the survey JSON.
    note : str
        A placeholder output (not used in this implementation).
    """

    @component.output_types(documents=List[Document], note=str)
    def run(self, json_results: List[Dict]):
        """
        Processes survey JSON data and converts each question and its choices into a document.

        Parameters:
        ----------
        json_results : List[Dict]
            A list of dictionaries, where each dictionary represents a survey with questions as keys
            and a list of choices as values.

        Returns:
        -------
        dict
            A dictionary containing a list of Document objects under the key 'documents'.
        """
        all_documents = []  # Initialize a list to hold all documents

        # Iterate over each JSON file (survey data) in the input list
        for survey_data in json_results:
            documents = []  # List to hold the documents for the current survey

            # Iterate through each question and its choices in the survey data
            for question, choices in survey_data.items():
                # Join each choice's values to form a text block for the choices
                choices_text = '\n'.join([', '.join(choice) for choice in choices])
                
                # Combine the question and its associated choices into a single text content
                content = f"{question}\n{choices_text}"

                # Create a Haystack Document object with the content
                document = Document(content=content)
                documents.append(document)  # Add the document to the current survey's document list

            # Add the documents from the current survey to the overall document list
            all_documents.extend(documents)

        # Return the list of all documents as output
        return {"documents": all_documents}

def indexing_pipeline_builder(document_store: QdrantDocumentStore, cohere_key: str) -> Pipeline:
    """
    Builds and returns an indexing pipeline for processing survey JSON files, embedding the content,
    and storing it in a QdrantDocumentStore.

    Parameters:
    ----------
    document_store : QdrantDocumentStore
        An initialized QdrantDocumentStore for storing the embedded documents.
    cohere_key : str
        API key for the Cohere API used to generate embeddings.

    Returns:
    -------
    Pipeline
        A configured indexing pipeline that processes JSON survey data, generates embeddings, and stores them.
    """
    
    # Initialize the indexing pipeline
    indexing_pipeline = Pipeline()
    
    # Set up the embedder using Cohere's API for embedding multilingual documents
    embedder = CohereDocumentEmbedder(api_key=Secret.from_token(cohere_key), model="embed-multilingual-v3.0")
    
    # Set up the document writer, which writes the documents to the document store
    writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)
    
    # Add components to the indexing pipeline
    # Custom component that processes JSON files and generates a document for each survey question
    indexing_pipeline.add_component(name="SurveyJSONToDocument", instance=SurveyJSONToDocument())
    
    # Add the embedder to the pipeline for embedding the processed documents
    indexing_pipeline.add_component("embedder", embedder)
    
    # Add the document writer to save the embedded documents into the QdrantDocumentStore
    indexing_pipeline.add_component("writer", writer)
    
    # Connect the components to form a data flow
    # Output from SurveyJSONToDocument is passed to the embedder
    indexing_pipeline.connect("SurveyJSONToDocument.documents", "embedder")
    
    # Output from the embedder is passed to the document writer
    indexing_pipeline.connect("embedder", "writer")
    
    return indexing_pipeline

# ***************************************************Create retriever pipeline***************************************************
def retriever_pipeline_builder(document_store: QdrantDocumentStore, cohere_key: str, groq_api: str, groq_key: str) -> Pipeline:
    """
    Builds a retriever-augmented generation (RAG) pipeline for document-based Q&A. The pipeline:
    - Embeds the input text
    - Retrieves relevant documents from a QdrantDocumentStore
    - Constructs a prompt with the retrieved documents
    - Generates a response using an LLM model
    
    Parameters:
    ----------
    document_store : QdrantDocumentStore
        An initialized QdrantDocumentStore where documents are stored and retrieved from.
    cohere_key : str
        API key for Cohere, used to generate embeddings for the input text.
    groq_api : str
        API base URL for the Groq LLM service.
    groq_key : str
        API key for accessing the Groq LLM service.
    
    Returns:
    -------
    Pipeline
        A configured pipeline for document retrieval and question answering with the LLM.
    """

    # Initialize the OpenAIGenerator (LLM) component for generating responses
    llm = OpenAIGenerator(
        api_key=Secret.from_token(groq_key),
        api_base_url=groq_api,
        model="llama3-8b-8192",
        generation_kwargs={
            "max_tokens": 512,        # Limit the response length to 512 tokens
            "temperature": 0          # Set temperature to 0 for deterministic responses
        }
    )

    # Define the prompt template for generating responses based on retrieved documents
    template = """
    Role:
        You are an AI designed to answer questions strictly based on the information contained in the provided documents. 
        Your task is to analyze the available content and generate a relevant response.
    Guidelines:
        * Greetings: If the input is a greeting, respond with a brief, friendly message, such as: "Hello! How can I assist you today?"
        * No Documents Available: If no documents are provided or they are empty, respond with: "I don't have the necessary information to answer your question."
        * Token Limit for Greetings/No Documents/General Question: Keep your response under 30 tokens if no documents are available or if the input is a greeting or the question is not explicitly covered by the documents.
        * Document-Based Answers: For questions, base your response only on the content from the provided documents. Avoid using any external knowledge.
        * Insufficient Information: If the documents don't contain the necessary information to answer the question, respond with: "I don't have the information."

    Example Scenarios:
        * Greeting Example:
            Input: "Hi there!"
            Output: "Hello! How can I assist you today?"
        * No Documents Example:
            Input: "What is the capital of France?"
            Output: "I don't have the necessary information to answer your question."
        * Document-Based Answer Example:
            Input: "What is the main ingredient in a Caesar salad?"
            Output: (Based on the content of the provided documents)
        * General Question Example:
            Input: "What is the difference between euro and dollar?"
            Output: "I don't have the information."
    
    Important Notes:
        * Ensure that all responses are accurate and relevant to the content provided in the documents.
        * Do not use any external knowledge unless explicitly stated in the provided documents.
        * If multiple documents are provided, loop through them to formulate the answer based on the combined content.
    
    \nDocuments:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}
    
    \nQuestion: {{ question }}?
    """

    # Initialize the PromptBuilder with the defined template
    prompt_builder = PromptBuilder(template=template)

    # Initialize the Cohere text embedder component for embedding the input text
    embedder = CohereTextEmbedder(api_key=Secret.from_token(cohere_key), model="embed-multilingual-v3.0")

    # Initialize the Qdrant retriever for fetching relevant documents based on the query embedding
    retriever = QdrantEmbeddingRetriever(document_store=document_store, top_k=2)

    # Initialize the pipeline for RAG (Retriever-Augmented Generation)
    rag_pipeline = Pipeline()

    # Add components to the pipeline
    rag_pipeline.add_component("text_embedder", embedder)  # Embed the input text
    rag_pipeline.add_component("retriever", retriever)     # Retrieve documents based on the embedding
    rag_pipeline.add_component("prompt_builder", prompt_builder)  # Build the prompt with retrieved documents
    rag_pipeline.add_component("llm", llm)                 # Generate the final answer using the LLM

    # Connect components in the pipeline
    rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")  # Pass embedding to retriever
    rag_pipeline.connect("retriever", "prompt_builder.documents")                 # Pass retrieved documents to the prompt builder
    rag_pipeline.connect("prompt_builder", "llm")                                 # Pass prompt to the LLM for response generation

    return rag_pipeline

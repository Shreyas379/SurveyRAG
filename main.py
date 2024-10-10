import base64
import tempfile
import argparse
import io
import os
import streamlit as st

from utils import (document_store_init, 
indexing_pipeline_builder,
retriever_pipeline_builder,
pipeline_process_file_and_convert_to_json_records
)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run the Streamlit app.')
    parser.add_argument('--environment', 
                        type=str, 
                        choices=['local', 'cloud'], 
                        default='cloud',
                        help='Specify the environment: "local" or "cloud".')
    args = parser.parse_args()
    
    if args.environment == 'cloud':
        # Access secret values
        cohere_api_key = st.secrets["COHERE_API_KEY"]
        groq_api = st.secrets["GROQ_API"]
        groq_key = st.secrets["GROQ_KEY"]
        qdrant_api = st.secrets["QDRANT_API"]
        qdrant_key = st.secrets["QDRANT_KEY"]
    else:
        from dotenv import load_dotenv
        load_dotenv()
        # Access secret values
        cohere_api_key = os.getenv("COHERE_API_KEY")
        groq_api = os.getenv("GROQ_API")
        groq_key = os.getenv("GROQ_KEY")
        qdrant_api = os.getenv("QDRANT_API")
        qdrant_key = os.getenv("QDRANT_KEY")        

    # Store secrets in session_state
    #cohere
    if "cohere_api_key" not in st.session_state:
        st.session_state.cohere_api_key = cohere_api_key
    #groqcloud
    if "groq_api" not in st.session_state:
        st.session_state.groq_api = groq_api
    if "groq_key" not in st.session_state:
        st.session_state.groq_key = groq_key
    #Qdrant
    if "qdrant_api" not in st.session_state:
        st.session_state.qdrant_api = qdrant_api
    if "qdrant_key" not in st.session_state:
        st.session_state.qdrant_key = qdrant_key

    #initialize document store
    document_store = document_store_init(api_key=qdrant_key, url=qdrant_api)

    #retriever_pipeline
    retriever_pipeline = retriever_pipeline_builder(document_store=document_store, 
                                                    cohere_key=st.session_state.cohere_api_key,
                                                    groq_api = st.session_state.groq_api,
                                                    groq_key=st.session_state.groq_key)
    # sidebar section
    with st.sidebar:
        # Toggle pop-up state
        if "show_instructions" not in st.session_state:
            st.session_state.show_instructions = False
        
        # Button to toggle instructions
        if st.button("ℹ️ Show/Hide Instructions"):
            st.session_state.show_instructions = not st.session_state.show_instructions

        if st.session_state.show_instructions:
            st.markdown("""
                <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px;">
                
                ### 📄 Step 1: Upload Your excel files and Index Them
                - **Upload Options:** Drag and drop your files or click on **'Browse Files'** to select them. (Maximum file size: **200MB**).
                - **Start Indexing:** Click the **'📄 Index Document'** button to initiate the indexing process.
                
                ### 💬 Step 2: Ask Your Questions
                -  Use the text box below to type your question. 📝
                </div>
                """, unsafe_allow_html=True)
        st.subheader("Upload your excel files and start interacting with the chatbot 🤖.")
        st.image('https://cdn-icons-png.flaticon.com/512/2037/2037149.png')
        uploaded_files = st.file_uploader("### 🗂️ Upload Excel files", type=["xlsx", "xls"], accept_multiple_files=True)
        if st.button('📄 Index Documents'):
            if uploaded_files is not None:
                try:
                    result_file_1, result_file_2 = pipeline_process_file_and_convert_to_json_records(uploaded_files)
                    
                    #indexing pipeline 
                    indexing_pipeline = indexing_pipeline_builder(document_store=document_store, cohere_key=st.session_state.cohere_api_key)
                    indexing_pipeline.run({"SurveyJSONToDocument":{'json_results':[result_file_1,result_file_2]}})
                    st.success('Document indexed, you can start asking questions!', icon="✅")
                
                except Exception as e:
                    st.warning(f'Failed to index document, please try again \n error:{str(e)}', icon="⚠️")

            else:
                st.warning("Please upload the documents and provide the missing fields.", icon="⚠️")
    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)

    #Main page
    st.title("Survey Insights RAG 📊🌍")


    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [] 


    st.image('https://cdn.dribbble.com/users/689802/screenshots/4044319/respage_chatbot_bubbletalk_dribbble.gif')
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate bot response
        response = retriever_pipeline.run({"text_embedder":{"text": prompt}, "prompt_builder": {"question": prompt}})

        # Add bot message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response["llm"]["replies"][0]})
        # Display bot message in chat message container
        with st.chat_message("assistant"):
            st.markdown(response["llm"]["replies"][0])
if __name__ == "__main__":
    main()

import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from qdrant_class import QdrantInsertRetrievalAll
from dotenv import load_dotenv
import tempfile
import os, uuid

load_dotenv()

st.set_page_config(page_title="DocumentGPT", page_icon=":💬:", layout="wide")
st.header("DocumentGPT 📚💬")

# Initialize URL and API Key for Qdrant
url = os.getenv("QDRANT_URL")
api_key = os.getenv("QDRANT_API_KEY")

# Initialize Qdrant handler
qdrant_handler = QdrantInsertRetrievalAll(api_key=api_key, url=url)

# Function to load and split PDF file into pages with metadata
def get_pdf_text(file_path, file_name):
    loader = PyMuPDFLoader(file_path=file_path)
    pages = loader.load_and_split()
    # Add metadata to each page
    for i, page in enumerate(pages):
        page.metadata.update({
            "source": file_name,
            "page": i + 1,
            "file_type": "PDF"
        })
    return pages

# Function to load and split text file with metadata
def get_txt_text(file_path, file_name):
    loader = TextLoader(file_path)
    splits = loader.load_and_split()
    # Add metadata to each split
    for i, split in enumerate(splits):
        split.metadata.update({
            "source": file_name,
            "section": i + 1,
            "file_type": "TXT"
        })
    return splits

# Sidebar to upload files
with st.sidebar:
    uploaded_files = st.file_uploader("Upload your file", type=['pdf'], accept_multiple_files=True)
    if st.button("Process"):
        Documents = []
        
        if uploaded_files:
            st.write("Files Loaded Splitting...")
            for uploaded_file in uploaded_files:
                try:
                    os.makedirs('data', exist_ok=True)
                    file_path = f"data/{uploaded_file.name}{uuid.uuid1()}"
                    with open(file_path, 'wb') as fp:
                        fp.write(uploaded_file.read())

                    split_tup = os.path.splitext(uploaded_file.name)
                    file_extension = split_tup[1]

                    if file_extension == ".pdf":
                        Documents.extend(get_pdf_text(file_path, uploaded_file.name))
                    elif file_extension == ".txt":
                        Documents.extend(get_txt_text(file_path, uploaded_file.name))

                except Exception as e:
                    st.error(f"Error processing this file: {uploaded_file.name} {e}")
                finally:
                    os.remove(file_path)
        else:
            st.error("No file uploaded.")

        if Documents:
            # Set collection name and store in session
            collection_name = os.path.splitext(uploaded_file.name)[0]
            st.session_state["collection_name"] = collection_name
            
            # Clear chat history when new file is uploaded
            if "langchain_messages" in st.session_state:
                st.session_state["langchain_messages"] = []
            
            st.write("Indexing Please Wait...")
            
            try:
                # Initialize embeddings
                embeddings = HuggingFaceEmbeddings(model_name="distiluse-base-multilingual-cased-v1")
                
                # Insert documents using the QdrantHandler
                qdrant = qdrant_handler.insertion(Documents, embeddings, collection_name)
                
                st.write("Indexing Done")
                st.success(f"Documents from {uploaded_file.name} added to collection '{collection_name}'")
                st.session_state["processtrue"] = True
                
            except Exception as e:
                st.error(f"Error indexing: {e}")

if "processtrue" in st.session_state:
    from chat import main
    main()
else:
    st.info("Please Upload Your Files.")

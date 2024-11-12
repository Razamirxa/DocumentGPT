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
st.header("📚 DocumentGPT 💬")
# Custom CSS
st.markdown("""
    <style>
        /* Main container styling */
        .main {
            padding: 2rem;
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e9f2 100%);
        }
        
        /* Header styling */
        .stHeader {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }
        
        /* Custom header with icon */
        .custom-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            font-size: 2.5rem;
            font-weight: bold;
            color: #1e3a8a;
            margin-bottom: 1rem;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* File uploader styling */
        .stFileUploader {
            border: 2px dashed #cbd5e1;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .stFileUploader:hover {
            border-color: #3b82f6;
            background-color: #f8fafc;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #3b82f6;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
            margin-top: 1rem;
        }
        
        .stButton > button:hover {
            background-color: #2563eb;
            transform: translateY(-2px);
        }
        
        /* Success message styling */
        .success-message {
            background-color: #dcfce7;
            color: #166534;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #22c55e;
            margin: 1rem 0;
        }
        
        /* Error message styling */
        .error-message {
            background-color: #fee2e2;
            color: #991b1b;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #ef4444;
            margin: 1rem 0;
        }
        
        /* Info message styling */
        .info-message {
            background-color: #dbeafe;
            color: #1e40af;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #3b82f6;
            margin: 1rem 0;
        }
        
        /* Chat container styling */
        .chat-container {
            background-color: white;
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-top: 2rem;
        }
        
        /* Chat message styling */
        .chat-message {
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 8px;
            max-width: 80%;
        }
        
        .user-message {
            background-color: #e9ecef;
            margin-left: auto;
        }
        
        .bot-message {
            background-color: #f8f9fa;
            margin-right: auto;
        }
    </style>
""", unsafe_allow_html=True)



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
    st.markdown('<div class="sidebar-container">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload your file", type=['pdf'], accept_multiple_files=True)
    if st.button("Process"):
        Documents = []
        
        if uploaded_files:
            st.markdown('<div class="info-message">Files Loaded. Splitting...</div>', unsafe_allow_html=True)

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
                st.markdown('<div class="info-message">Indexing Please Wait...</div>', unsafe_allow_html=True)
            
            st.write("Indexing Please Wait...")
            
            try:
                # Initialize embeddings
                embeddings = HuggingFaceEmbeddings(model_name="distiluse-base-multilingual-cased-v1")
                
                # Insert documents using the QdrantHandler
                qdrant = qdrant_handler.insertion(Documents, embeddings, collection_name)
                st.markdown('<div class="success-message">Indexing Done!</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="success-message">Documents from {uploaded_file.name} added to collection \'{collection_name}\'</div>', unsafe_allow_html=True)
                
                st.write("Indexing Done")
                st.success(f"Documents from {uploaded_file.name} added to collection '{collection_name}'")
                st.session_state["processtrue"] = True
                
            except Exception as e:
                st.markdown(f'<div class="error-message">Error indexing: {e}</div>', unsafe_allow_html=True)
                

if "processtrue" in st.session_state:
    from chat import main
    main()
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="info-message">Please Upload Your Files.</div>', unsafe_allow_html=True)





# DocumentGPT 📚💬

## Overview
DocumentGPT is a Streamlit-based application designed to help users upload and analyze PDF and text documents. Using Qdrant for vector storage and Google LLM for question-answering, DocumentGPT provides responses to user queries from the content of uploaded files, with embedded references for each answer.

## Features
- Upload and process PDF and text documents
- Chunked document indexing with metadata in Qdrant vector storage
- Search and retrieve relevant information with Google’s language models
- Citation of sources in response to queries

## Installation

### Requirements
- Python 3.7 or above
- Access to a Qdrant instance and Google LLM API keys

### Steps
1. Clone this repository:
    ```bash
    https://github.com/Razamirxa/DocumentGPT.git
    cd DocumentGPT
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file to store your API keys:
    ```plaintext
    QDRANT_URL=<your-qdrant-url>
    QDRANT_API_KEY=<your-qdrant-api-key>
    GOOGLE_API_KEY=<your-google-api-key>
    ```

4. Run the app:
    ```bash
    streamlit run home.py
    ```

## Usage
- Upload one or more PDF or text files using the sidebar.
- After processing, ask questions about the content, and DocumentGPT will retrieve relevant information along with cited sources.

## License
This project is open-source, licensed under MIT.

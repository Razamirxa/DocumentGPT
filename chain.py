from operator import itemgetter
from typing import List, Tuple
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from qdrant_class import QdrantInsertRetrievalAll
import streamlit as st
import os

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# First, define the template constants
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template  = """If the input is a greeting like "hi", "hello", or "hey":
    Greet the user warmly and ask how you can assist them.

If the input is unrelated to the provided context:
    Respond with "I'm sorry, I don't have enough information to answer that question based on the given context."

Otherwise:
    Answer the question based only on the following context and maintain a friendly tone.
    
<context>
{context}
</context>

Question: {question}

Sources: For each source, include the document name and page/section number.
"""

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", template),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{question}")
])

def get_vectorstore():
    """Create a new vectorstore instance using the current collection name"""
    # Initialize URL and API Key for Qdrant
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    
    embeddings = HuggingFaceEmbeddings(model_name="distiluse-base-multilingual-cased-v1")
    
    collection_name = st.session_state.get("collection_name", "default_collection")
    
    qdrant_handler = QdrantInsertRetrievalAll(api_key=api_key, url=url)
    return qdrant_handler.retrieval(collection_name, embeddings)

def _combine_documents(docs):
    # Extract source information while combining documents
    sources = []
    content = []
    
    for doc in docs:
        # Handle the document content directly without JSON serialization
        content.append(doc.page_content)
        
        # Extract source information from metadata
        if hasattr(doc, 'metadata'):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', doc.metadata.get('section', 'N/A'))
            sources.append(f"{source} (Page/Section {page})")
    
    # Combine content and add sources at the end
    combined_text = "\n\n".join(content)
    if sources:
        sources = list(set(sources))  # Remove duplicates
        combined_text += "\n\nSources:\n" + "\n".join(sources)
    
    return combined_text

def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    return chat_history

def create_chain():
    """Create a new chain instance with the current vectorstore"""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={'k': 15})

    _search_query = RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            RunnablePassthrough.assign(
                chat_history=lambda x: _format_chat_history(x["chat_history"])
            )
            | CONDENSE_QUESTION_PROMPT
            | GoogleGenerativeAI(model="gemini-1.5-flash",temperature=0,api_key=GOOGLE_API_KEY) 
            | StrOutputParser(),
        ),
        RunnableLambda(itemgetter("question"))
    )

    _inputs = RunnableParallel({
        "question": lambda x: x["question"],
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
        "context": _search_query | retriever | _combine_documents,
    })

    return _inputs | ANSWER_PROMPT | GoogleGenerativeAI(
        model="gemini-1.5-pro",
        api_key=GOOGLE_API_KEY
    ) | StrOutputParser()

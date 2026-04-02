import sys
import os

from langchain_community.vectorstores import FAISS
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from chain.chain_builder import qa_chain

def create_faiss_index(chunks, embedding_model):
    """
    Create a FAISS index from text chunks using the specified embedding mo

    :param chunks: List of text chunks
    :param embedding_model: The embedding model to use
    :return: FAISS index
    """
    # Use the FAISS library to create an index from the provided text chun
    return FAISS.from_texts(chunks, embedding_model)

def perform_similarity_search(faiss_index, query, k=3):
    """
    Search for specific queries within the embedded transcript using the FAISS index.

    :param faiss_index: The FAISS index containing embedded text chunks
    :param query: The text input for the similarity search
    :param k: The number of similar results to return (default is 3)
    :return: List of similar results
    """
    # Perform the similarity search using the FAISS index
    results = faiss_index.similarity_search(query, k=k)
    return results

def retrieve(query, faiss_index, k=7):
    """
    Retrieve relevant context from the FAISS index based on the user's query.

    :param query: The user's query string.
    :param faiss_index: The FAISS index containing the embedded documents.
    :param k: The number of most relevant documents to retrieve (default is 7).
    :return: A list of the k most relevant documents (or document chunks).
    """
    relevant_context = faiss_index.similarity_search(query, k=k)
    return relevant_context


def generate_answer(question, faiss_index, qa_chain, k=7):
    """
    Retrieve relevant context and generate an answer based on user input.
    Args:
    question: str
    The user's question.
    faiss_index: FAISS
    The FAISS index containing the embedded documents.
    qa_chain: LLMChain
    The question-answering chain (LLMChain) to use for generating
    k: int, optional (default=3)
    The number of relevant documents to retrieve.
    Returns:
    str: The generated answer to the user's question.
    """
    # Retrieve relevant context
    relevant_context = retrieve(question, faiss_index, k=k)
    # Generate answer using the QA chain
    answer = qa_chain.invoke({"context": relevant_context, "question": question})
    return answer

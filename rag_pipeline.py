import os
from typing import Optional, List

from pdfminer.high_level import extract_text
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

import google.generativeai as genai
from byaldi import RAGMultiModalModel
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
import numpy as np
import json

from mistralai import Mistral
os.environ["TOKENIZERS_PARALLELISM"] = "false"

GOOGLE_API_KEY = "AIzaSyDuDWi3ZMs4p2G08HzlIX0aMjAEtinUWP8"
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-2.0-flash-001')

def chunk_and_retrieve():
    RAW_KNOWLEDGE_BASE = []
    root_path = "eecs-487-docs"
    data_paths = ["487w25-syllabus.pdf", "14-llm.pdf", "13-transformer (1).pdf", "1-introduction.pdf", "2-textnormalization.pdf", "3-languagemodel (1).pdf", "4-textcategorization-nb (1).pdf", "5-mlbasics (2).pdf", "6-parsing_part1 (1).pdf","7-parsing_part2 (1).pdf", "8-semantics (1).pdf", "9-neurallm (1).pdf", "10-svd (1).pdf",  "11-rnn-languagemodel (1).pdf", "12-rnn-lstm (1).pdf", "15-summarization.pdf"]    
    full_data_paths = [f"{root_path}/{path}" for path in data_paths]
    data_paths = full_data_paths
    
    for data_path in data_paths:
        RAW_KNOWLEDGE_BASE.append(LangchainDocument(extract_text(data_path)))

    # added natural questions to knowledge base
    # jsonl_docs = read_jsonl("NQ-open.train.jsonl")
    # RAW_KNOWLEDGE_BASE.extend(jsonl_docs)

    EMBEDDING_MODEL_NAME = "thenlper/gte-small"

    def split_documents(
        chunk_size: int,
        knowledge_base: List[LangchainDocument],
        tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
    ) -> List[LangchainDocument]:
        """
        Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
        """
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(tokenizer_name),
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size / 10),
            add_start_index=True,
            strip_whitespace=True,
        )

        docs_processed = []
        for doc in knowledge_base:
            docs_processed += text_splitter.split_documents([doc])

        # Remove duplicates
        unique_texts = {}
        docs_processed_unique = []
        for doc in docs_processed:
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                docs_processed_unique.append(doc)

        return docs_processed_unique


    docs_processed = split_documents(
        512,  # We choose a chunk size adapted to our model
        RAW_KNOWLEDGE_BASE,
        tokenizer_name=EMBEDDING_MODEL_NAME,
    )

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=False,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
    )

    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )

    return embedding_model, KNOWLEDGE_VECTOR_DATABASE, docs_processed

def retrieve_info(user_query, embedding_model, KNOWLEDGE_VECTOR_DATABASE, docs_processed):
    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
    docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE)
    
    retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=7)
    return retrieved_docs

def generate_rag_response(query, retrieved_docs, model):
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
    You are a helpful assistant that answers questions primarily using the provided context. 
    If the question can't be answered from the context, try to answer without the context using your own knowledge
    If the question is straightforward or based on common knowledge, you may respond without the given context. 
    Prioritize the context when it is applicable, conversational answer is more appropriate.
    
    CONTEXT: {context}
    
    QUESTION: {query}

    ANSWER:
    """
    config = {
        "temperature": 1,
        "top_p": 0.85,
        "top_k": 2,
        "max_output_tokens": 200
    }
    
    response = model.generate_content(prompt, generation_config=config)

    return response.text


def convert_pdfs_to_images(pdf_folder):
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    # pdf_files = [pdf_folder]
    all_images = {}

    for doc_id, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        images = convert_from_path(pdf_path)
        all_images[doc_id] = images

    return all_images

def rag_images(text_query):
    all_images = convert_pdfs_to_images("data/")
    # docs_retrieval_model = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", device="cpu")
    # docs_retrieval_model = RAGMultiModalModel.from_index("14-index", device="cpu")
    docs_retrieval_model = RAGMultiModalModel.from_index("stored_indicies", device="cpu")

    results = docs_retrieval_model.search(text_query, k=3)
    top_image_score = results[0]['score']


    def get_grouped_images(results, all_images):
        grouped_images = []

        for result in results:
            doc_id = result['doc_id']
            page_num = result['page_num']
            grouped_images.append(all_images[doc_id][page_num - 1])
        return grouped_images

    grouped_images = get_grouped_images(results, all_images)

    return grouped_images, top_image_score

def get_top_image(group_images):
    return np.array(group_images[0])

def generate_caption(img, query):
    context = img
    prompt = f"""
    Generate a brief and specific image caption that directly addresses the query. Your caption should:
    - Be no longer than 1 sentences

    CONTEXT: {context}

    QUESTION: {query}

    ANSWER:
    """
    config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 1,
        "max_output_tokens": 50
    }
    
    response = model.generate_content(prompt, generation_config=config)
    return response.text


def read_jsonl(file_path):
    """Read JSONL file and convert each entry to a LangchainDocument"""
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            # Adapt these fields based on your NQ-open format
            question = data.get('question', '')
            answers = data.get('answer', [])
            
            # Create content by combining question and answers
            content = f"Question: {question}\nAnswers: {', '.join(answers)}"
            
            # Create metadata from other fields if needed
            metadata = {
                "source": file_path,
                "type": "qa_pair",
                # Add other metadata fields you want to track
            }
            
            # Create LangchainDocument
            doc = LangchainDocument(page_content=content, metadata=metadata)
            documents.append(doc)
    
    return documents


MISTRAL_API_KEY = "sChtPgP9W9GBeOeJGvKPK0LRQVgXAXMK"
mistral_client = Mistral(api_key=MISTRAL_API_KEY)
mistral_model = "mistral-large-latest"

def mistral_generate_rag_response(query, retrieved_docs, mistral_model):
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
    You are a helpful assistant that answers questions primarily using the provided context. 
    If the question can't be answered from the context, try to answer without the context using your own knowledge
    If the question is straightforward or based on common knowledge, you may respond without the given context. 
    Prioritize the context when it is applicable, conversational answer is more appropriate.
    
    CONTEXT: {context}
    
    QUESTION: {query}

    ANSWER:
    """
    
    response = mistral_client.chat.complete(
        model=mistral_model,
        messages = [
            {
                "role":"user",
                "content": prompt
            },
        ],
        max_tokens=250 # set to prevent against exploit
    )

    return response.choices[0].message.content

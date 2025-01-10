import os
import json
import re
from langchain_milvus import Milvus
from langchain.schema import Document
from dotenv import load_dotenv
from uuid import uuid4
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
import nltk
import copy
import fitz

load_dotenv()


# def extract_bold_to_bold_text(pdf_path):
#     doc = fitz.open(pdf_path)
#     bold_to_bold_texts = []
#     current_text = ""

#     for page_num in range(doc.page_count):
#         page = doc.load_page(page_num)
#         blocks = page.get_text("dict")["blocks"]
        
#         for block in blocks:
#             if block['type'] == 0:  # This means it's a text block
#                 for line in block["lines"]:
#                     for span in line["spans"]:
#                         text = span["text"]
                        
#                         # Check if the text is bold
#                         if "bold" in span["font"] or span["font"].startswith("bold"):
#                             if current_text:  # If there's text collected, save the current sentence
#                                 bold_to_bold_texts.append(current_text.strip())
#                             current_text = text  # Start a new bold-to-bold collection
#                         else:
#                             current_text += " " + text  # Add normal text to the current collection

#                 # At the end of the block, check if there is any leftover text
#                 if current_text.strip():
#                     bold_to_bold_texts.append(current_text.strip())
#                     current_text = ""  # Reset for next section

#     return bold_to_bold_texts




def extract_sentences_from_pdf(pdf_file_path):
    # Open the PDF file
    with open(pdf_file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        
        # Extract text from each page
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()

    # Use NLTK's sentence tokenizer to split text into sentences
    sentences = nltk.sent_tokenize(text)
    
    # Filter out sentences that don't end with a period
    sentences = [sentence for sentence in sentences if sentence.endswith('.')]
    
    return sentences





def extract_article(sentences):
    processed_data = []
    
    # for sentence in sentences:
    #     if 'Chương' in sentence:
    #         pattern_Chapter = r"Chương (\d+|[IVXLCDM]+)(.*?\n.*?)(?:\n|$)"
    #         match = re.findall(pattern_Chapter, sentence)
    #         current['Chapter'] = f'Chương {match[0][0]} {match[0][1].strip()}'
    #         processed_data.append(copy.copy(current))
    while True:
        current = {'page_content': '', 'Chapter': '', 'Article': ''}
        for sentence in sentences:
            if 'Chương' in sentence and 'Điều' in sentence:
                pattern_Chapter = r"Chương (\d+|[IVXLCDM]+)(.*?\n.*?)(?:\n|$)"
                match_C = re.findall(pattern_Chapter, sentence)
                current['Chapter'] = f'Chương {match_C[0][0]} {match_C[0][1].strip()}'
                pattern_Article = r"Điều \d+[a-zA-Z]?"
                match_A = re.findall(pattern_Article, sentence)
                current['Article'] = f'Điều {match_A[0]}'
            if 'Chương' in sentence:
                pattern_Chapter = r"Chương (\d+|[IVXLCDM]+)(.*?\n.*?)(?:\n|$)"
                match = re.findall(pattern_Chapter, sentence)
                current['Chapter'] = f'Chương {match[0][0]} {match[0][1].strip()}'
            if 'Điều' in sentence and len(sentence) <= 150:
                pattern_Article = r"Điều \d+[a-zA-Z]?"
                match = re.findall(pattern_Article, sentence)
                if match:
                    current['Article'] = f'{match[0]}'
            else:
                current['page_content'] = sentence
            processed_data.append(copy.copy(current))
        
        break    
                
    return processed_data


# def load_data_from_local(filename: str, directory: str) -> tuple:
#     file_path = f"{directory}/{filename}"
#     # loader = PyPDFLoader(file_path)
#     loader = pdfplumber.open(file_path)
#     # data = loader.load_and_split()
#     # data = loader.load()
#     data = loader.pages[0]
#     clean_text = data.filter(lambda obj: obj["object_type"] == "char" and "Bold" in obj["fontname"])
#     clean_text = clean_text.extract_text()


#     return clean_text, filename.rsplit('.', 1)[0].replace('_', ' ')

def seed_milvus(URI_link: str, collection_name: str, filename: str, directory: str) -> Milvus:
    embeddings = OllamaEmbeddings(
        model="llama3.1" 
    )

    
    # local_data, doc_name = load_data_from_local(filename, directory)
    bolds = extract_sentences_from_pdf(f"{directory}/{filename}")
    processed = extract_article(bolds)
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # documents = text_splitter.split_documents(local_data)
    # chunks = chunking(local_data)
    # documents = [Document(page_content=chunk) for chunk in chunks]
    
    uuids = [str(uuid4()) for _ in range(len(processed))]
    documents = [
        Document(
            page_content= doc['page_content'] or '',
            metadata = {
                'Chapter' : doc['Chapter'] or '',
                'Article' : doc['Article'] or '',
            }
        )   
        for doc in processed
    ]
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI_link},
        collection_name=collection_name,
        drop_old=True  
    )

    vectorstore.add_documents(documents=documents, ids=uuids)
    print('vector: ', vectorstore)
    return vectorstore
    # return vectorstore


def connect_to_milvus(URI_link: str, collection_name: str) -> Milvus:
    embeddings = OllamaEmbeddings(model="llama3.1")
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI_link},
        collection_name=collection_name,
    )
    return vectorstore

def main():

    seed_milvus('http://localhost:19530', 'data_test', 'CS311_law.pdf', 'data')

if __name__ == "__main__":
    main()
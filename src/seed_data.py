import re
from langchain_milvus import Milvus
from typing import List
from langchain.schema import Document
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from uuid import uuid4
import copy
import fitz
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
load_dotenv()
torch.cuda.empty_cache() 

def extractArticle(pdf_path):
    doc = fitz.open(pdf_path)
    Article = []
    current_text = ""

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if block['type'] == 0:  # This means it's a text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"]
                        
                        if "bold" in span["font"] or span["font"].startswith("bold"):
                            if current_text: 
                                Article.append(current_text.strip())
                            current_text = text  
                        else:
                            current_text += " " + text 


                if current_text.strip():
                    Article.append(current_text.strip())
                    current_text = "" 

    return Article

    
class EmbeddingFunctionWrapper:
    def __init__(self, model):
        self.model = SentenceTransformer(model, trust_remote_code=True)

    def embed_documents(self, texts, batch_size=8):
        # embeddings = []
        # for i in range(0, len(texts), batch_size):
        #     batch_texts = texts[i:i + batch_size]
        #     tokenized_texts = [tokenize(text) for text in batch_texts]
        #     embeddings.extend(self.model.encode(tokenized_texts))
        #     torch.cuda.empty_cache()  # Clear GPU cache
        # return embeddings
        embeddings = []
        for text in texts:
            embeddings.append(self.model.encode(text))
            torch.cuda.empty_cache()
        return embeddings

    def embed_query(self, text):
        # You can also add this if needed
        # tokenized_text = tokenize(text)
        
        return self.model.encode([text])[0]



def loadEmbeddingModel():
    return EmbeddingFunctionWrapper('dangvantuan/vietnamese-embedding-LongContext')

embeddings = loadEmbeddingModel()

def extractSentencesFromPdf(pdf_path):
    sentences = []
    
    # Open the PDF file
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            # Get the text of the page
            page_text = doc[page_num].get_text()
            
            # Split text into sentences using a regex
            page_sentences = re.split(r'(?<=[.!?]) +', page_text)
            sentences.extend(page_sentences)
    
    return sentences


def mergeArticle(texts):
    processed = []
    current = ""
    pattern_Article = r"Điều \d+[a-zA-Z]?\."
    current_article = None
    for text in texts:
        match = re.search(pattern_Article, text)
        if match != None:
            processed.append({
                    'page_content': copy.copy(current.strip()),
                    'article': copy.copy(current_article) or 'None'
            })
            current_article = match.group(0)
            current = text
        else :
            if current == "":
                current = text
            else:
                current += " " + text
    if current_article is not None:
        processed.append({
            'page_content': copy.copy(current.strip()),
            'article': copy.copy(current_article)
    })
    return processed


def chunkToSmallerText(items):
    chunked_strings = []
    separators = [". \n"]
    text_splitter = RecursiveCharacterTextSplitter(separators=separators, chunk_size=500, chunk_overlap=50)
    for item in items:
        chunks = text_splitter.split_text(item['page_content'])
        for chunk in chunks:
            chunked_strings.append(
                {
                    'page_content': copy.copy(chunk),
                    'article': copy.copy(item['article'])
                }
            )
    for chunk in chunked_strings:
        chunk['page_content'] = re.sub(r'\s+', ' ', chunk['page_content']).strip()
    return chunked_strings

def printResult(chunks):
    for chunk in chunks:
        print(chunk)
        print("---------------")

def seed_milvus(URI_link: str, collection_name: str, file_name: str, directory: str) -> Milvus:
    data = extractSentencesFromPdf(f"{directory}/{file_name}")
    data = mergeArticle(data)
    data = chunkToSmallerText(data)
    printResult(data)
    # documents = [
    #     Document(
    #         page_content= doc['page_content'] or '',
    #         metadata = {
    #             'article': doc['article'] or ''
    #         }
    #     )
    #     for doc in data
    # ]
    # uuids = [str(uuid4()) for _ in range(len(documents))]

    # vectorstore = Milvus(
    #     embedding_function= embeddings,
    #     connection_args={"uri": URI_link},
    #     collection_name=collection_name,
    #     drop_old=True  
    # )

    # vectorstore.add_documents(documents=documents, ids=uuids)
    # print('vector: ', vectorstore)
    # return vectorstore



def connect_to_milvus(URI_link: str, collection_name: str) -> Milvus:
    embeddings_model = embeddings
    vectorstore = Milvus(
        embedding_function=embeddings_model,
        connection_args={"uri": URI_link},
        collection_name=collection_name,
    )
    return vectorstore

def main():

    seed_milvus('http://localhost:19530', 'data_test', 'legal-document.pdf', 'data')

if __name__ == "__main__":
    main()
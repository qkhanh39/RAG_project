from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from seed_data import seed_milvus, connect_to_milvus
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
#=======================
from transformers import GenerationConfig, TextStreamer
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig




def load_llm(): 
    llm = ChatOllama(
        model="llama3.1", 
        temperature=0,
        streaming=True
    )
    return llm

def create_prompt(template):
    prompt = PromptTemplate(template = template, input_variables=["context", "question"])
    return prompt

def create_retriver(vectorstore, collection_name: str = "data_test"):
    
    milvus_retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 8}
    )
    
    documents = [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in vectorstore.similarity_search("", k=100)
    ]
    
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 8

    ensemble_retriever = EnsembleRetriever(
            retrievers=[milvus_retriever, bm25_retriever],
            weights=[0.8, 0.2]
    )
    return ensemble_retriever


def create_qa_chain(prompt, llm, vectorstore):
    ensenble_retriever = create_retriver(vectorstore)
    
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= "stuff",
        retriever = ensenble_retriever,
        return_source_documents = True,
        chain_type_kwargs= {'prompt': prompt}
    )
    return llm_chain




def initialize_chain(collection_name: str = "data_test"):
    vectorstore = connect_to_milvus('http://localhost:19530', collection_name)

    llm = load_llm()
    # template = """<|im_start|>system
    # Chỉ được sử dụng thông tin dưới đây để trả lời. Không được đưa ra câu trả lời dựa trên kiến thức bên ngoài. Nếu không biết, hãy trả lời 'Tôi không biết'.
    # {context}<|im_end|>
    # <|im_start|>user
    # {question}<|im_end|>
    # <|im_start|>assistant"""
    template = (
        "### System:\n"
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:\n{output}"
    )

    # template = """"
    # Sử dụng thông tin đã cho để trả lời các câu hỏi.
    # Nếu thông tin không đủ để đưa ra câu trả lời chính xác, hãy trả lời tôi không biết.
    # Context: {context}
    # Question: {question}
    # """
    prompt = create_prompt(template)
    llm_chain  = create_qa_chain(prompt, llm, vectorstore)
    
    return llm_chain
    # return llm_chain, vectorstore
    
def main():
    vectorstore = connect_to_milvus('http://localhost:19530', collection_name="data_test")
    llm = ChatOllama(
            model="llama3.1", 
            temperature=0,
            streaming=True
        )
    def expand_query(query, llm):
        prompt = f"Expand this query with related terms:\n{query}"
        expanded_query = llm.predict(prompt)
        return expanded_query

    # Mở rộng và tìm kiếm
    query = "Điều 6 là gì?"
    expanded_query = expand_query(query, llm)
    print(expanded_query)
    results = vectorstore.similarity_search(expanded_query, k=2)
    print("<====================>")
    print(query)

if __name__ == "__main__":
    print("hi")
    main()
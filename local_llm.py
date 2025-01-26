from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from seed_data import connect_to_milvus
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

def load_llm():
    llm = ChatOllama(
        model="llama3.1", 
        temperature=0,
        streaming=True
    )
    return llm


def create_prompt():
    # Phân chia template thành các loại tin nhắn
    # system_message = SystemMessagePromptTemplate.from_template(
    #     """Below is an instruction that describes a task, paired with an input that provides further context.
    #     Your task is to answer the user's query using only the retrieved information from the database.
    #     Do not provide answers based on assumptions or external knowledge, only use the information in the context. 
    #     If the retrieved information does not contain enough details, respond with 'I don't know'.
    #     Please answer in Vietnamese.
    #     Prioritize the top context which relevent to the question. \n
    #     Context information is below.
    #     ---------------------
    #     {context}
    #     ---------------------"""
    # )
    
    system_message = SystemMessagePromptTemplate.from_template(
    """
    Bạn là một chuyên gia về luật giao thông đường bộ ở Việt Nam.
    Dưới đây là câu hỏi về một điều luật cùng với ngữ cảnh cung cấp thông tin liên quan.
    Nhiệm vụ của bạn là trả lời câu hỏi bằng cách sử dụng duy nhất thông tin được lấy từ ngữ cảnh.
    1. Nếu ngữ cảnh có chứa thông tin chính xác, hãy trả lời cụ thể, đầy đủ, và trực tiếp dựa trên thông tin đó.
    2. Nếu ngữ cảnh không cung cấp đủ thông tin để trả lời câu hỏi, hãy trả lời: "Tôi không biết".
    3. Luôn ưu tiên sử dụng ngữ cảnh liên quan trực tiếp đến câu hỏi và tránh suy diễn hoặc đưa ra câu trả lời ngoài phạm vi ngữ cảnh.
    Ngữ cảnh thông tin:
    {context}
    Yêu cầu trả lời:
    1) Trả lời ngắn gọn, đúng trọng tâm.
    2) Sử dụng ngôn ngữ tiếng Việt.
    
    """
    )
    user_message = HumanMessagePromptTemplate.from_template(
        """
        Đây là câu hỏi cần trả lời dựa trên thông tin trong ngữ cảnh đã cung cấp:
        Câu hỏi:
        {question}
    
        Lưu ý:
        - Hãy trả lời dựa trên ngữ cảnh, không sử dụng kiến thức bên ngoài.
        - Nếu không có đủ thông tin trong ngữ cảnh, hãy trả lời: "Tôi không biết".
        """
    )
    prompt = ChatPromptTemplate.from_messages([system_message, user_message])
    return prompt


def create_retriver(vectorstore, collection_name: str = "data_test"):
    
    milvus_retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 2, "include_metadata": True}
    )
    
    documents = [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in vectorstore.similarity_search("", k=100)
    ]
    
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 1
    
    
    
    ensemble_retriever = EnsembleRetriever(
            retrievers=[milvus_retriever, bm25_retriever],
            weights=[0.7, 0.3]
    )
    
    
    return ensemble_retriever



def create_qa_chain(prompt, llm, vectorstore):
    ensemble_retriever = create_retriver(vectorstore)
   
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= "stuff",
        return_source_documents = True,
        chain_type_kwargs= {'prompt': prompt},
        retriever=ensemble_retriever
    )
    return llm_chain


def initialize_chain(collection_name: str = "data_test"):
    vectorstore = connect_to_milvus('http://localhost:19530', collection_name)

    llm = load_llm()
    
    prompt = create_prompt()
    llm_chain  = create_qa_chain(prompt, llm, vectorstore)
    
    return llm_chain
    # return llm_chain, vectorstore
    



# def query_and_print(chain, vectorstore, question: str):
#     create_retriver(vectorstore)
#     response = chain.invoke({"query": question})
#     print(f"\nLLM Response: {response['result']}")
#     print("================")


# # Initialize the chain and database
# llm_chain, vectorstore = initialize_chain(collection_name="data_test")


# question = "Điều 10 là gì ?"
# query_and_print(llm_chain, vectorstore, question)

# for i in range (1, 64):
#     question = f"Điều {i} là gì ?"
#     query_and_print(llm_chain, vectorstore, question)











































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

def load_llm():
    llm = ChatOllama(
        model="llama3.1", 
        temperature=0,
        streaming=True
    )
    return llm

def creat_prompt(template):
    prompt = PromptTemplate(template = template, input_variables=["context", "question"])
    return prompt


def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= "stuff",
        retriever = db.as_retriever(search_kwargs = {"k":3}),
        return_source_documents = True,
        chain_type_kwargs= {'prompt': prompt}
    )
    return llm_chain


def initialize_chain(collection_name: str = "data_test"):
    
    
    db = connect_to_milvus('http://localhost:19530', collection_name)
    llm = load_llm()
    template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không tìm thấy câu trả lời ở thông tin tôi cung cấp, hãy nói không biết, đừng cố tạo ra câu trả lời\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
    prompt = creat_prompt(template)
    llm_chain  = create_qa_chain(prompt, llm, db)
    
    
    return llm_chain, db


def query_and_print(chain, db, question: str, top_k: int = 3):
    # Use the retriever directly to get the top-k vectors
    retriever = db.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(question)
    
    print(f"\nTop-{top_k} Retrieved Documents:")
    for i, doc in enumerate(docs, start=1):
        print(f"\nDocument {i}:")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
    
    # Run the question through the QA chain
    print("\nRunning the QA Chain...")
    response = chain.invoke({"query": question})
    print(f"\nLLM Response: {response}")



# Initialize the chain and database
llm_chain, db = initialize_chain(collection_name="data_test")

# Ask a question and print results
question = "Có bao nhiêu loại tài khoản số đẹp?"
query_and_print(llm_chain, db, question, top_k=3)






































# def get_retriever(collection_name: str = "data_test") -> EnsembleRetriever:
#     """
#     Tạo một ensemble retriever kết hợp vector search (Milvus) và BM25
#     Args:
#         collection_name (str): Tên collection trong Milvus để truy vấn
#     Returns:
#         EnsembleRetriever: Retriever kết hợp với tỷ trọng:
#             - 70% Milvus vector search (k=4 kết quả)
#             - 30% BM25 text search (k=4 kết quả)
#     """
#     try:
#         # Kết nối với Milvus và tạo vector retriever
#         vectorstore = connect_to_milvus('http://localhost:19530', collection_name)
#         milvus_retriever = vectorstore.as_retriever(
#             search_type="similarity", 
#             search_kwargs={"k": 4}
#         )

#         # Tạo BM25 retriever từ toàn bộ documents
#         documents = [
#             Document(page_content=doc.page_content, metadata=doc.metadata)
#             for doc in vectorstore.similarity_search("", k=20)
#         ]
        
#         if not documents:
#             raise ValueError(f"Không tìm thấy documents trong collection '{collection_name}'")
            
#         bm25_retriever = BM25Retriever.from_documents(documents)
#         bm25_retriever.k = 4

#         # Kết hợp hai retriever với tỷ trọng
#         ensemble_retriever = EnsembleRetriever(
#             retrievers=[milvus_retriever, bm25_retriever],
#             weights=[0.7, 0.3]
#         )
#         return ensemble_retriever
        
#     except Exception as e:
#         print(f"Lỗi khi khởi tạo retriever: {str(e)}")
#         # Trả về retriever với document mặc định nếu có lỗi
#         default_doc = [
#             Document(
#                 page_content="Có lỗi xảy ra khi kết nối database. Vui lòng thử lại sau.",
#                 metadata={"source": "error"}
#             )
#         ]
#         return BM25Retriever.from_documents(default_doc)


# def get_llm_and_agent(retriever):
#     """
#     Khởi tạo LLM và agent với Ollama
#     """
#     # Tạo retriever tool
#     tool = create_retriever_tool(
#         retriever,
#         "find_documents",
#         "Search for information of traffic law."
#     )
#     tools = [tool]
#     # Khởi tạo ChatOllama
#     llm = ChatOllama(
#         model="llama3.1",  # hoặc model khác tùy chọn
#         temperature=0,
#         streaming=True
#     )
 

#     # Thiết lập prompt template
#     system = """You are an expert at road traffic law. Your name is ChatchatAI. For road traffic law questions call the 'find_document' tool to answer"""
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", system),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("human", "{input}"),
#         MessagesPlaceholder(variable_name="agent_scratchpad"),
#         ("tool_names", "{tool_names}"),  # Include tool_names
#         ("tools", "{tools}")
#     ])

#     # Tạo agent
#     agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
#     return AgentExecutor(agent=agent, tools=tools, verbose=True)


# # Khởi tạo retriever và agent với collection mặc định
# retriever = get_retriever()
# agent_executor = get_llm_and_agent(retriever)
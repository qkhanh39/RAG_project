# RAG_project


<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->
<!-- code_chunk_output -->

* Contents:
    * [Introduction](#introduction)
    * [Repository Structure](#repository-structure)
    * [Setup Instructions](#setup-instructions)
    * [Usage](#usage)
<!-- /code_chunk_output -->

## Introduction
This project was created to manage street legal documents in Vietnam for the CS211 course during the Fall 2024 semester at the University of Information Technology, VNUHCM.

## Repository structure
```txt
RAG_Project
├── src
│   ├── data
│   ├── local_llm.py
│   ├── main.py
│   └── seed_data.py  
├── .env
├── README.md
├── docker-compose.yml
├── requirements.txt

```

## Setup instructions
1. Create a virtual environment to run this project
    - Run this conda command: 
    ```bash
    conda create -n [YOUR-ENV-NAME] python==3.8.18
    conda activate [YOUR-ENV-NAME]
    pip install -r requirements.txt 
    ```
2. Download the LLM model
In this project, we used LLaMa 3.1, to install it:
    - Visit this page: https://ollama.com/download
    - Choose the version suitable for your operating system.
    - Install according to the instructions.
    - Run this command: 
    ```bash
    ollama run llama3.1
    ```

3. Install and run Milvus database
    - Open your terminal at the project and run the below command for the first time:
    ```bash
    docker compose up --build
    ```
    - Next time, you only need to run: 
    ```bash
    docker compose upmilvus server
    ```
    - If you want to view the data seed in the Milvus: 
        - Run any command to find your IP address, for Ubuntu: 
        ```bash
        ifconfig
        ``` 
        - Then, run: 
        ```bash
        docker run -p 8000:3000 -e MILVUS_URL={YOUR IP}:19530 zilliz/attu:v2.4
        ```

4. Set up to check the result of retriever and LLM model.
    - Visit this page and create your API key: https://smith.langchain.com/
    - Create a .env file and add 4 lines below: 
        - LANGCHAIN_TRACING_V2=true
        - LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
        - LANGCHAIN_API_KEY="your-langchain-api-key-here"
        - LANGCHAIN_PROJECT="project-name"

5. Seed data into Milvus
    - Run this command to seed data into Milvus: 
    ```python
    python seed_data.py
    ```
### Usage
Now you can run and interact with the chatbot by running this command:
    ```python
    streamlit run main.py
    ```
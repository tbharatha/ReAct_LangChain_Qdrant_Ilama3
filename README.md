## Integrating ReAct LangChain with Qdrant and Illama3: A Comprehensive Guide

### Introduction

Combining cutting-edge tools and models is crucial for delivering efficient query handling and insightful responses in the world of natural language processing (NLP) and information retrieval. This tutorial explores how three powerful technologies — LangChain’s ReAct Agents, the Qdrant Vector Database, and the Llama3 large language model (LLM) from the Groq endpoint — can work together to supercharge intelligent information retrieval systems.

Fact: Agents are nothing but LLMs with a complex prompt.

### Source
The documents referred to in this article include:
- [Google Terms of Service](https://policies.google.com/terms?hl=en-US)
- [OpenAI Terms of Use](https://openai.com/policies/terms-of-use/)
- [Facebook Terms of Service](https://www.facebook.com/legal/terms?paipv=0&eav=AfYU7-7Cf-zij8FiJxMbZUIw3eF6mj9sXRTd01_PiZSBjEuKOE3VHDVPzP31EkYsVZk&_rdr)
- [SLS Handbook - Contract Law](https://students.ucsd.edu/_files/sls/handbook/SLSHandbook-Contract_Law.pdf)

### Setting Up the Environment

Necessary packages:

- PyPDF2 -> Extracting Text from PDFs
- Qdrant -> Using Qdrant as Vector Database
- LangChain -> For Agents’ Creation and Text Processing
- ChatGroq -> For Providing API Endpoint for LLMs by Groq for Lightning Fast Inference
- Gradio -> UI for the Chat Interface

```python
import os
from PyPDF2 import PdfReader
import numpy as np
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
import numpy as np
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import Tool
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.llms import OpenAI
from langchain_groq import ChatGroq
import gradio as gr
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
```

Groq API for inference: [Groq API Keys](https://console.groq.com/keys)

Qdrant API (local or cloud): [Qdrant Cloud](https://cloud.qdrant.io/login)

### Data Preparation

Extract text from PDF documents stored in a specified directory:

```python
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    extracted_text = ""
    for page in reader.pages:
        extracted_text += page.extract_text()
    return extracted_text

def extract_text_from_pdfs_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            extracted_text = extract_text_from_pdf(pdf_path)
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_filepath = os.path.join(directory, txt_filename)
            with open(txt_filepath, "w") as txt_file:
                txt_file.write(extracted_text)

directory_path = "Docs/"
extract_text_from_pdfs_in_directory(directory_path)
```

Store document data:

```python
directory_path = "Docs"
txt_files = [file for file in os.listdir(directory_path) if file.endswith('.txt')]

all_documents = {}
for txt_file in txt_files:
    loader = TextLoader(os.path.join(directory_path, txt_file))
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator="\n")
    docs = text_splitter.split_documents(documents)
    for doc in docs:
        doc.metadata["source"] = txt_file  # Add source metadata
    all_documents[txt_file] = docs
```

### Storing Data in Qdrant Vector DB Using Custom Embeddings

Use "all-mpnet-base-v2" embedding model from HuggingFace:

```python
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

qdrant_collections = {}
for txt_file in txt_files:
    qdrant_collections[txt_file] = Qdrant.from_documents(
        all_documents[txt_file],
        embeddings,
        location=":memory:", 
        collection_name=txt_file,
    )

retriever = {}
for txt_file in txt_files:
    retriever[txt_file] = qdrant_collections[txt_file].as_retriever()
```

### Setting ReAct Agents

Define functions for agents to call and retrieve additional information:

```python
def get_relevant_document(name : str) -> str:
    search_name = name
    best_match = process.extractOne(search_name, txt_files, scorer=fuzz.ratio)
    selected_file = best_match[0]
    selected_retriever = retriever[selected_file]
    global query
    results = selected_retriever.get_relevant_documents(query)
    global retrieved_text
    total_content = "\n\nBelow are the related document's content: \n\n"
    chunk_count = 0
    for result in results:
        chunk_count += 1
        if chunk_count > 4:
            break
        total_content += result.page_content + "\n"
    retrieved_text = total_content
    return total_content

def get_summarized_text(name : str) -> str:
    from transformers import pipeline
    summarizer = pipeline("summarization", model="Falconsai/text_summarization")
    global retrieved_text
    article = retrieved_text
    return summarizer(article, max_length=1000, min_length=30, do_sample=False)[0]['summary_text']

def get_today_date(input : str) -> str:
    import datetime
    today = datetime.date.today()
    return f"\n {today} \n"

def get_age(name: str, person_database: dict) -> int:
    if name in person_database:
        return person_database[name]["Age"]
    else:
        return None

def get_age_info(name: str) -> str:
    person_database = {
        "Sam": {"Age": 21, "Nationality": "US"},
        "Alice": {"Age": 25, "Nationality": "UK"},
        "Bob": {"Age": 11, "Nationality": "US"}
    }
    age = get_age(name, person_database)
    if age is not None:
        return f"\nAge: {age}\n"
    else:
        return f"\nAge Information for {name} not found.\n"
```

Wrap the functions around `Tool` for agent usage:

```python
get_age_info_tool = Tool(
    name="Get Age",
    func=get_age_info,
    description="Useful for getting age information for any person. Input should be the name of the person."
)

get_today_date_tool = Tool(
    name="Get Todays Date",
    func=get_today_date,
    description="Useful for getting today's date"
)

get_relevant_document_tool = Tool(
    name="Get Relevant document",
    func=get_relevant_document,
    description="Useful for getting relevant document that we need."
)

get_summarized_text_tool = Tool(
    name="Get Summarized Text",
    func=get_summarized_text,
    description="Useful for getting summarized text for any document."
)
```

### Setting Agent Prompts

Using LangChain Hub:

```python
prompt_react = hub.pull("hwchase17/react")
```

### ReAct Agent Creation

Create and configure a ReAct agent for intelligent query processing:

```python
tools = [get_relevant_document_tool, get_summarized_text_tool, get_today_date_tool, get_age_info_tool]
retrieved_text = ""

model = ChatGroq(model_name="llama3-70b-8192", groq_api_key=GROQ_API_KEY, temperature=0)
react_agent = create_react_agent(model, tools=tools, prompt=prompt_react)
react_agent_executor = AgentExecutor(
    agent=react_agent, tools=tools, verbose=True, handle_parsing_errors=True
)
```

### Executing Complex Queries

Example queries:

1. Summary of age requirements for OpenAI Services:
```python
query = “Give me the summary for the question: What age requirement is specified for using the OpenAI Services, and what provision applies if the user is under 18?”
react_agent_executor.invoke({"input": query})
```

2. Resources Google offers for user assistance:
```python
query = “Give me summary of What resources does Google offer to users for assistance and guidance in using its services?”
react_agent_executor.invoke({"input": query})
```

3. Eligibility for OpenAI Services in 2027:
```python
query = “I am Bob. Will I be eligible in 2027 for the age requirement specified for using the OpenAI Services by OpenAI Terms?”
react_agent_executor.invoke({"input": query})
```

### Conclusion

The integration of LangChain’s ReAct Agents with Qdrant and Llama3 enhances query handling efficiency and provides insightful responses to user queries. This step-by-step guide helps build an intelligent information retrieval system using these advanced technologies.

You can also add a Gradio UI over the system:

```python
import gradio as gr
from io import StringIO
import sys
import re

def generate_response(question):
    tools = [get_age_info_tool, get_health_info_tool]
    model = ChatGroq(model_name

="llama3-70b-8192", groq_api_key=GROQ_API_KEY, temperature=0)
    react_agent = create_react_agent(model, tools=tools, prompt=prompt_react)
    react_agent_executor = AgentExecutor(
        agent=react_agent, tools=tools, verbose=True, handle_parsing_errors=True
    )
    response = react_agent_executor.invoke({"input": question})
    return response["output"]

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    question = gr.Textbox(placeholder="Ask a question about any topic")
    submit_button = gr.Button("Submit")
    submit_button.click(fn=generate_response, inputs=question, outputs=chatbot)

demo.launch(share=True)
```

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

Sure, I'll update the directory path to "dataset" and provide a `requirements.txt` file along with an improved `README.md`.

### Updated Code with Directory Path Change

#### Extracting Text from PDFs

```python
def extract_text_from_pdfs_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            extracted_text = extract_text_from_pdf(pdf_path)
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_filepath = os.path.join(directory, txt_filename)
            with open(txt_filepath, "w") as txt_file:
                txt_file.write(extracted_text)

directory_path = "dataset/"
extract_text_from_pdfs_in_directory(directory_path)
```

#### Storing Document Data

```python
directory_path = "dataset"
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

### Requirements File (`requirements.txt`)

```
PyPDF2
numpy
langchain
qdrant-client
sentence-transformers
transformers
fuzzywuzzy
gradio
torch
```

### Improved README File (`README.md`)

```markdown
# Intelligent Information Retrieval with LangChain ReAct Agents, Qdrant, and Llama3

This project demonstrates how to integrate LangChain's ReAct Agents, Qdrant Vector Database, and the Llama3 model to build a robust intelligent information retrieval system. The setup allows users to query and retrieve information efficiently from a collection of documents.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Running the Gradio Interface](#running-the-gradio-interface)

## Prerequisites

Ensure you have the following installed:
- Python 3.7 or higher
- pip

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/yourproject.git
   cd yourproject
   ```

2. Install the required Python packages:
   ```sh
   pip install -r requirements.txt
   ```

## Data Preparation

1. Place your PDF documents in the `dataset` directory.
2. Run the script to extract text from the PDFs:
   ```python
   def extract_text_from_pdfs_in_directory(directory):
       # Code to extract text
       ...

   directory_path = "dataset/"
   extract_text_from_pdfs_in_directory(directory_path)
   ```

3. Store the extracted document data:
   ```python
   directory_path = "Dataset"
   txt_files = [file for file in os.listdir(directory_path) if file.endswith('.txt')]

   all_documents = {}
   for txt_file in txt_files:
       # Code to store document data
       ...
   ```

## Usage

1. Define functions for the ReAct agents to call and retrieve additional information:
   ```python
   def get_relevant_document(name: str) -> str:
       # Code to get relevant document
       ...

   def get_summarized_text(name: str) -> str:
       # Code to get summarized text
       ...

   def get_today_date(input: str) -> str:
       # Code to get today's date
       ...

   def get_age_info(name: str) -> str:
       # Code to get age information
       ...
   ```

2. Wrap the functions around `Tool` for agent usage:
   ```python
   get_age_info_tool = Tool(
       name="Get Age",
       func=get_age_info,
       description="Useful for getting age information for any person. Input should be the name of the person."
   )
   ```

3. Set up the ReAct agents using LangChain Hub:
   ```python
   prompt_react = hub.pull("hwchase17/react")
   tools = [get_relevant_document_tool, get_summarized_text_tool, get_today_date_tool, get_age_info_tool]

   model = ChatGroq(model_name="llama3-70b-8192", groq_api_key=GROQ_API_KEY, temperature=0)
   react_agent = create_react_agent(model, tools=tools, prompt=prompt_react)
   react_agent_executor = AgentExecutor(
       agent=react_agent, tools=tools, verbose=True, handle_parsing_errors=True
   )
   ```

4. Execute complex queries:
   ```python
   query = "Give me the summary for the question: What age requirement is specified for using the OpenAI Services, and what provision applies if the user is under 18?"
   response = react_agent_executor.invoke({"input": query})
   ```

## Running the Gradio Interface

1. Create a Gradio UI for the system:
   ```python
   import gradio as gr

   def generate_response(question):
       tools = [get_age_info_tool, get_health_info_tool]
       model = ChatGroq(model_name="llama3-70b-8192", groq_api_key=GROQ_API_KEY, temperature=0)
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

2. Run the script to launch the interface:
   ```sh
   python script_name.py
   ```

Now you have a fully functional intelligent information retrieval system with a user-friendly interface.

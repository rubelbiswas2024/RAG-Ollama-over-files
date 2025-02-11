# RAG Ollama over files through VPN server

This project implements a Retrieval-Augmented Generation (RAG) system over files such as PDFs, Word documents, and spreadsheets. Users can upload a document, input a query, and get answers based on the document's content.

## Features
- Supports PDF, DOCX, XLSX, XLS, and CSV files.
- Uses FAISS for indexing and retrieval.
- Uses the Qwen2.5:32b model locally via Ollama.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rag-over-files.git
   cd rag-over-files

Install dependencies:
pip install -r requirements.txt

Run the application:
python main.py

Requirements:
Python 3.8+
Local Ollama setup


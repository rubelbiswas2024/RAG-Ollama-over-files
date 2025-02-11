import gradio as gr
import shutil
import os
import pandas as pd
import PyPDF2
import docx
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import subprocess

# Initialize Ollama client parameters
OLLAMA_HOST = 'http://localhost:11434'
MODEL_NAME = 'qwen2.5:32b'

# Function to handle file upload and return the file path
def upload_file(file):
    UPLOAD_FOLDER = "./data"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    destination = os.path.join(UPLOAD_FOLDER, os.path.basename(file.name))
    shutil.copy(file.name, destination)
    return destination

# Function to extract text based on file type
def extract_text_from_file(file_path):
    try:
        if file_path.endswith(".pdf"):
            reader = PyPDF2.PdfReader(file_path)
            return "".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif file_path.endswith(".docx"):
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        elif file_path.endswith((".xlsx", ".xls", ".csv")):
            df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)
            return df.to_string()
        else:
            return "Unsupported file format. Please upload a PDF, DOCX, XLSX, XLS, or CSV file."
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# Function to split text into smaller chunks
def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.create_documents([text])

# Create embeddings and index chunks with FAISS
def create_faiss_index(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    chunk_texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(chunk_texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, model

# Retrieve relevant chunks from FAISS
def retrieve_relevant_chunks(faiss_index, query, model, chunks, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# Query the Ollama model locally
def query_ollama(model_name, input_text):
    try:
        result = subprocess.run(
            ['ollama', 'run', model_name, input_text],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        return f"Error querying Ollama: {str(e)}"

# Function to process the RAG pipeline
def rag_pipeline(file_path, query):
    text = extract_text_from_file(file_path)
    if "Error" in text:
        return text
    chunks = split_text_into_chunks(text)
    faiss_index, model = create_faiss_index(chunks)
    relevant_chunks = retrieve_relevant_chunks(faiss_index, query, model, chunks)
    context = "\n".join([chunk.page_content for chunk in relevant_chunks])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    return query_ollama(MODEL_NAME, prompt)

# Gradio UI components
with gr.Blocks(css=".center-text {text-align: center;}") as demo:
    with gr.Row():
        gr.Markdown(
            """
            # RAG over Files
            Upload your documents and ask a question. This system processes the file and provides answers based on its content.
            """, elem_id="title"
        )

    file_input = gr.File(label="Add your documents! Provide a file path (PDF, DOCX, XLSX, XLS, or CSV)")
    file_path_output = gr.Textbox(label="File Path", interactive=False, visible=False)
    file_content_output = gr.Textbox(label="File Content", interactive=False)
    query_input = gr.Textbox(label="Query:", placeholder="What's up?")
    answer_output = gr.Textbox(label="Answer", interactive=False)
    clear_button = gr.Button("Clear All ↺")

    # File input triggers content extraction
    file_input.change(
        lambda file: (upload_file(file), extract_text_from_file(upload_file(file))),
        inputs=[file_input],
        outputs=[file_path_output, file_content_output]
    )

    # Query input triggers the RAG pipeline
    query_input.submit(
        lambda path, query: rag_pipeline(path, query),
        inputs=[file_path_output, query_input],
        outputs=[answer_output]
    )

    # Clear button resets the outputs
    clear_button.click(
        lambda: (None, None, None),
        inputs=[],
        outputs=[file_path_output, file_content_output, answer_output]
    )

demo.launch(share=True)

import os
import numpy as np
import pymupdf
import pytesseract
from PIL import Image
import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from gpt4all import GPT4All

# Prevent CPU thread thrashing
os.environ["OMP_THREAD_LIMIT"] = "1"

# Configuration
MODEL_PATH = "model.gguf"  # Path to your downloaded GGUF file
EMBED_MODEL_NAME = "BAAI/bge-m3"

print("Loading local embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

print("Loading local LLM via GPT4All...")
# Load GGUF model directly without C++ compilation requirements
llm = GPT4All(MODEL_PATH, allow_download=False)

# Global variables for vector database
current_faiss_index = None
current_chunks = []

def extract_text_from_pdf(pdf_file_path):
    """Extracts text from PDF, falling back to OCR if page has no text stream."""
    doc = pymupdf.open(pdf_file_path)
    extracted_text = ""
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()
        
        # Fallback to OCR if page is image-based or vectorized
        if not text:
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img, lang="ara+eng").strip()
            
        if text:
            extracted_text += f"\n--- Page {page_num + 1} ---\n" + text
            
    doc.close()
    return extracted_text

def process_uploaded_pdf(file_obj):
    """Processes uploaded PDF file into chunked embeddings and FAISS index."""
    global current_faiss_index, current_chunks
    
    if file_obj is None:
        return "Please upload a valid PDF file."

    pdf_path = file_obj.name
    print(f"Processing uploaded file: {pdf_path}")
    
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text.strip():
        return "Failed to extract text from the PDF."

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", " ", ""]
    )
    current_chunks = text_splitter.split_text(raw_text)
    
    embeddings = embed_model.encode(current_chunks, normalize_embeddings=True)
    dimension = embeddings.shape[1]
    
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(embeddings).astype(np.float32))
    current_faiss_index = index

    return f"PDF processed successfully! Created {len(current_chunks)} chunks."

def query_pdf_local(query_text, task_mode):
    """Retrieves context locally and generates response using local GPT4All LLM."""
    global current_faiss_index, current_chunks

    if current_faiss_index is None or not current_chunks:
        return "Please upload and process a PDF document first.", ""

    if not query_text.strip():
        return "Please enter a valid prompt.", ""

    # 1. Local Semantic Search via FAISS
    query_vector = embed_model.encode([query_text], normalize_embeddings=True)
    distances, indices = current_faiss_index.search(np.array(query_vector).astype(np.float32), k=3)
    
    retrieved_chunks = [current_chunks[i] for i in indices[0] if i < len(current_chunks)]
    context_str = "\n\n---\n\n".join(retrieved_chunks)

    # 2. Prompt Construction
    if task_mode == "Summarize":
        system_prompt = "Summarize the provided text context clearly based on the user request."
    else:
        system_prompt = "Answer the user question directly using only the provided context."

    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\nContext:\n{context_str}\n\nTask: {query_text}<|im_end|>\n<|im_start|>assistant\n"

    # 3. Local LLM Generation
    with llm.chat_session():
        generated_text = llm.generate(prompt, max_tokens=512, temp=0.3)

    return generated_text.strip(), context_str

# Gradio Web Interface
with gr.Blocks(title="Local PDF RAG Assistant") as demo:
    gr.Markdown("# Fully Local PDF Document Q&A & Summarization")
    gr.Markdown("Upload any PDF file. Embeddings, vector search, and LLM generation run 100% locally.")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload PDF Document", file_types=[".pdf"])
            process_btn = gr.Button("Process PDF", variant="primary")
            status_output = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            user_query = gr.Textbox(label="Question or Topic Prompt", placeholder="Enter your query...", lines=2)
            task_mode = gr.Radio(choices=["Question Answering", "Summarize"], value="Question Answering", label="Mode")
            query_btn = gr.Button("Submit Query", variant="secondary")
            
            answer_output = gr.Textbox(label="Local LLM Output", lines=8)
            context_output = gr.Accordion("Retrieved Context Chunks", open=False)
            context_text = gr.Textbox(label="Context", lines=8)

    process_btn.click(fn=process_uploaded_pdf, inputs=[file_input], outputs=[status_output])
    query_btn.click(fn=query_pdf_local, inputs=[user_query, task_mode], outputs=[answer_output, context_text])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
    
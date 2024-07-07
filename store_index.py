from src.helper import download_hugging_face_embeddings
from src.helper import Langchain_RAG
import os
embeddings = download_hugging_face_embeddings()
### Pdf file Path for RAG
pdf_file_path = "/home/praveent/End-To-End-Madical-Chatbot-NLP-Project-1/data/Medical_book.pdf"
retriever = Langchain_RAG(pdf_file_path=pdf_file_path)
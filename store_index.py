from src.helper import download_hugging_face_embeddings
from src.helper import Langchain_RAG
# from dotenv import load_dotenv
import os

# load_dotenv()
embeddings = download_hugging_face_embeddings()
### Pdf file Path for RAG
pdf_file_path = "/home/praveent/End-To-End-Madical-Chatbot-NLP-Project-1/data/Medical_book.pdf"
retriever = Langchain_RAG(pdf_file_path=pdf_file_path)

# from langchain_community.vectorstores import FAISS
# from langchain.docstore import InMemoryDocstore
# import faiss
# # embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
# # Initialize an empty FAISS index
# dimension = embeddings.client.get_sentence_embedding_dimension()
# index = faiss.IndexFlatL2(dimension)

# docstore = InMemoryDocstore()

# ### for semantic cache
# vector_store = FAISS(
#     embedding_function=embeddings,
#     index=index,
#     docstore=docstore,
#     index_to_docstore_id={}
# )
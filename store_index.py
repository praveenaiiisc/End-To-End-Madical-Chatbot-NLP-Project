# from src.helper import load_pdf, text_split, download_hugging_face_embeddings
# from langchain.vectorstores import Pinecone
# import pinecone
from src.helper import Langchain_RAG
from dotenv import load_dotenv
import os

load_dotenv()

# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

# extracted_data = load_pdf("data/")
# text_chunks = text_split(extracted_data)
# embeddings = download_hugging_face_embeddings()
### Pdf file Path for RAG
pdf_file_path = "/home/praveent/End-To-End-Madical-Chatbot-NLP-Project-1/data/Medical_book.pdf"
retriever = Langchain_RAG(pdf_file_path=pdf_file_path)

# #Initializing the Pinecone
# pinecone.init(api_key=PINECONE_API_KEY,
#               environment=PINECONE_API_ENV)


# index_name="medical-bot"

# #Creating Embeddings for Each of The Text Chunks & storing
# docsearch=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)

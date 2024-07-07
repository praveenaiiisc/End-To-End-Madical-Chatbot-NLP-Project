from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader,PDFMinerLoader,TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Extract data from pdf and split 
class Langchain_RAG:
    def __init__(self,pdf_file_path):
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.pdf_file_path = pdf_file_path
        print("loading pdf file , this may take time to process")
        self.loader = loader = PDFMinerLoader(self.pdf_file_path)   
        self.data = self.loader.load()
        print("<< pdf file loaded")
        print("<< chunking")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"])
        self.texts = text_splitter.split_documents(self.data)
        print("<< chunked")
        self.get_vec_value= FAISS.from_documents(self.texts, self.embeddings)
        print("<< saved")
        self.retriever = self.get_vec_value.as_retriever(search_kwargs={"k": 4}) 
    def __call__(self,query):
        rev = self.retriever.get_relevant_documents(query) 
        return "".join([i.page_content for i in rev])
    
#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return embeddings
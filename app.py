# # Install Pytorch & other libraries
# ! pip install "torch==2.1.2" tensorboard

# # Install Hugging Face libraries
# ! pip install  --upgrade \
#   "transformers==4.38.2" \
#   "datasets==2.16.1" \
#   "accelerate==0.26.1" \
#   "evaluate==0.4.1" \
#   "bitsandbytes==0.42.0" \
#   "trl==0.7.11" \
#   "peft==0.8.2" \
#     "langchain" \
# # "sentence-transformers" \
# "faiss-cpu"
# ! pip install unstructured
# ! pip install pdfminer
# ! pip install pdfminer.six
# ! pip install -U langchain-community 




from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from store_index import retriever

# from dotenv import load_dotenv
# # from src.prompt import *

import torch
from IPython.display import display_markdown
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
import transformers
import time
from langchain_community.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
import faiss
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
# Initialize an empty FAISS index
dimension = embeddings.client.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(dimension)

docstore = InMemoryDocstore()

### for semantic cache
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=docstore,
    index_to_docstore_id={}
)
app = Flask(__name__) # it will create the web application
##########################################################################################################################
##########################################################################################################################

# load_dotenv()
# Hugging Face model id
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    token='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',
    model_kwargs={
        "torch_dtype": torch.float16,
        # "quantization_config": {"load_in_4bit": True},
        "low_cpu_mem_usage": True,
    },
)

terminators =  [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]


import time
# This class is used to generate responses from an LLM model
class Llama3_8B_gen:
    def __init__(self, pipeline, embeddings, vector_store, threshold):
        self.pipeline = pipeline
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.threshold = threshold
        
    @staticmethod
    def generate_prompt(query,retrieved_text):    # Generate Prompt of my query data and recevied text
        messages = [
            {"role": "system", "content": "Answer the Question for the Given below context and information and not prior knowledge, only give the output result \n\ncontext:\n\n{}".format(retrieved_text) },
            {"role": "user", "content": query},]
        return pipeline.tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=True)
    
    def semantic_cache(self, query, prompt):                               # findout Similarity between query vectore and vectore_store
        query_embedding = self.embeddings.embed_documents([query])
        similar_docs = self.vector_store.similarity_search_with_score_by_vector(query_embedding[0], k=1) 
        
        if similar_docs and similar_docs[0][1] <self.threshold:            # if similarity less than my set thresold then return response from my cache data
            self.print_bold_underline("---->> From Cache")                 # otherwise generate text from pipeline and data to vectore_store
            return similar_docs[0][0].metadata['response']
        else:
            self.print_bold_underline("---->> From LLM")
            output = self.pipeline(prompt, max_new_tokens=512, eos_token_id=terminators, do_sample=True, temperature=0.7, top_p=0.9)
            
            response = output[0]["generated_text"][len(prompt):]
            self.vector_store.add_texts(texts = [query], 
                       metadatas = [{'response': response},])
            
            return response
            
    def generate(self, query, retrieved_context):
        start_time = time.time()                                          
        
        prompt = self.generate_prompt(query, retrieved_context)           # Generate prompt
        res = self.semantic_cache(query, prompt)                          # lokking Similarity and findout respose according to similarity from pipeline or verctore_sore(cache)
        
        end_time = time.time()
        execution_time = end_time - start_time                            # Claculate the running Time
        self.print_bold_underline(f"LLM generated in {execution_time:.6f} seconds")
        
        return res

    @staticmethod
    def print_bold_underline(text):
        print(f"\033[1m\033[4m{text}\033[0m")


text_gen = Llama3_8B_gen(pipeline=pipeline)

def Rag_qa(query):
    retriever_context = retriever(query)
    resut = text_gen.generate(query,retriever_context)
    return resut

##########################################################################################################################
##########################################################################################################################

# display_markdown(Rag_qa("What are Allergies"),raw=True)

# load_dotenv()

# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')


# embeddings = download_hugging_face_embeddings()

# #Initializing the Pinecone
# pinecone.init(api_key=PINECONE_API_KEY,
#               environment=PINECONE_API_ENV)

# index_name="medical-bot"

#Loading the index
# docsearch=Pinecone.from_existing_index(index_name, embeddings)


# PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# chain_type_kwargs={"prompt": PROMPT}

# llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
#                   model_type="llama",
#                   config={'max_new_tokens':512,
#                           'temperature':0.8})


# qa=RetrievalQA.from_chain_type(
#     llm=llm, 
#     chain_type="stuff", 
#     retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
#     return_source_documents=True, 
#     chain_type_kwargs=chain_type_kwargs)



@app.route("/")
def index():
    return render_template('chat.html')   # opend html cole on local host 



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    # print(input)
    print("user_input_query : ", input)
    # result=Rag_qa({"User_query": input})
    result=Rag_qa(input)
    print("Chatbot_Response : ", result)
    # print("Chatbot_Response : ", result["result"])
    return str(result)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)



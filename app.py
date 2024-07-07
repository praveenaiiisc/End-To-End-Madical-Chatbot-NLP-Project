from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from store_index import retriever
import torch
from IPython.display import display_markdown
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import pipeline
import transformers
import time
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import time

embeddings = download_hugging_face_embeddings()
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

# Hugging Face model id
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    token='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',
    model_kwargs={
        "torch_dtype": torch.float16,
        "quantization_config": {"load_in_4bit": True},
        "low_cpu_mem_usage": True,
    },
    device_map="cuda:2"
)

terminators =  [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

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


text_gen = Llama3_8B_gen(pipeline=pipeline,embeddings=embeddings,
                         vector_store=vector_store,threshold=0.1)

def Rag_qa(query):
    retriever_context = retriever(query)
    resut = text_gen.generate(query,retriever_context)
    return resut

##########################################################################################################################
##########################################################################################################################

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
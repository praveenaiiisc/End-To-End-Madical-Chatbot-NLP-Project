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
# from src.helper import download_hugging_face_embeddings
from store_index import retriever
# from langchain.vectorstores import Pinecone
# import pinecone
# from langchain.prompts import PromptTemplate
# from langchain.llms import CTransformers
# from langchain.chains import RetrievalQA
# from dotenv import load_dotenv
# from src.prompt import *
import os
import transformers
import torch
from IPython.display import display_markdown
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

app = Flask(__name__) # it will create the web application
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


class Llama3_8B_gen:
    def __init__(self,pipeline):
        self.pipeline= pipeline
        
    @staticmethod
    def generate_prompt(query,retrieved_text):
        messages = [
            {"role": "system", "content": "Answer the Question for the Given below context and information and not prior knowledge, only give the output result \n\ncontext:\n\n{}".format(retrieved_text) },
            {"role": "user", "content": query},]
        return pipeline.tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=True)
    
    def generate(self,query,retrieved_context):
        prompt = self.generate_prompt(query ,retrieved_context)
        output =  pipeline(prompt,max_new_tokens=512,eos_token_id=terminators,do_sample=True,
                            temperature=0.7,top_p=0.9,)         
        return output[0]["generated_text"][len(prompt):]


text_gen = Llama3_8B_gen(pipeline=pipeline)

def Rag_qa(query):
    retriever_context = retriever(query)
    resut = text_gen.generate(query,retriever_context)
    return resut

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



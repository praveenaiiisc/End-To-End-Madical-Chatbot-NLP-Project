# End-to-end-Medical-Chatbot-using-Llama2
- In this project, I implement a **MADICAL-CHATBOT** by using FLASK(Web framework) and FAISS(Building a vector storage) and LangChain i.e RAG(Retrieval Augmented Generation) implemantation to finetune model "meta-llama/Meta-Llama-3-8B-Instruct" on given dataset Madecal Book(GALE ENCYCLOPEDIA of MEDICINE) and ask question from this book, So this madical chatbot generate good respomce from this book.
- Result
- ![alt text](<Screenshot 2024-07-07 224124.png>)
- ![alt text](<Screenshot 2024-07-07 225942.png>) ![alt text](<Screenshot 2024-07-07 225900.png>) ![alt text](<Screenshot 2024-07-07 224409.png>) ![alt text](<Screenshot 2024-07-07 224325.png>) ![alt text](<Screenshot 2024-07-07 224247.png>)

###### ============================================================================================== #####
###### ============================= Descrived Implementation ======================================== #####
- 1. Clone github and create requirement.txt file and write every libary whatever you need in project.
- 2. make pinecone api key and pinecone api environment fron pinecone website for my data base
- 3. create data folder and put all data inthis folder, create model folder and download llama-2-7b-chat.ggmlv3.q4_0.bin model from haggingface and put in model folder, create reserch folder and make jyupitor notebook of 
- -----> first extract the data from pdf
- -----> after that create text chunk(split) from the extracted data
- -----> after that download the model of embedding from haggingface(this model create 384 dimention of vector)
- -----> make emebeddings of our data documents(pdf) and store in pinecone database
- 4. Creting Modular Coading Pipeline
- -------> for doing manually to create folder name we make a template.py file and created all folder using os and logging for info
- --------> for setup any file in local peackage so that i can easily import function from that file, for doing this we create a setup.py file (This is the main part of modular coading)
- --------> ok, Now goes to src/helper,py file and paste my code of documents(data) and emebeddings regarding and this file imported in store_index.py file
- 5.Now i have to complete frontend right now, So we create app.py and put model pipeline and question-answering regarding full code,
- ------>create one chatboot templte from html and css store in templtes and static folder, so just import in my app.py folder


###### =========================================================================== #####
###### =========================================================================== #####


#### How to run?
- Clone the repository

```bash
Project repo: https://github.com/
```

-  STEP 01- Create a conda environment after opening the repository

```bash
conda create -n mchatbot python=3.8 -y
```

```bash
conda activate mchatbot
```

- STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

```bash
# run the following command
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

```bash
open up localhost:
```

#### Techstack Used:
- Python
- LangChain
- Flask
- Meta Llama3
- FAISS
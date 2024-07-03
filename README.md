# End-to-end-Medical-Chatbot-using-Llama2

###### =========================================================================== #####
###### ============================= Note ======================================== #####
###### 1. Clone github and create requirement.txt file and write every libary whatever you need in project.
###### 2. make pinecone api key and pinecone api environment fron pinecone website for my data base
###### 3. create data folder and put all data inthis folder, create model folder and download llama-2-7b-chat.ggmlv3.q4_0.bin model from haggingface and put in model folder, create reserch folder and make jyupitor notebook of 
###### -----> first extract the data from pdf
###### -----> after that create text chunk(split) from the extracted data
###### -----> after that download the model of embedding from haggingface(this model create 384 dimention of vector)
###### -----> make emebeddings of our data documents(pdf) and store in pinecone database
###### 4. Creting Modular Coading Pipeline
###### -------> for doing manually to create folder name we make a template.py file and created all folder using os and logging for info
###### --------> for setup any file in local peackage so that i can easily import function from that file, for doing this we create a setup.py file (This is the main part of modular coading)
###### 5.


###### =========================================================================== #####
###### =========================================================================== #####


# How to run?
### STEPS:

Clone the repository

```bash
Project repo: https://github.com/
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n mchatbot python=3.8 -y
```

```bash
conda activate mchatbot
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
PINECONE_API_ENV = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


### Download the quantize model from the link provided in model folder & keep the model in the model directory:

```ini
## Download the Llama 2 Model:

llama-2-7b-chat.ggmlv3.q4_0.bin


## From the following link:
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
```

```bash
# run the following command
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:
```


### Techstack Used:

- Python
- LangChain
- Flask
- Meta Llama2
- Pinecone



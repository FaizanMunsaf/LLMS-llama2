# Import transformer classes for generaiton
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
# Import torch for datatype attributes
import torch
from flask import Flask, request
import locale
# Import the prompt wrapper...but for llama index
from llama_index.prompts.prompts import SimpleInputPrompt
# Import the llama index HF Wrapper
from llama_index.llms import HuggingFaceLLM

# Bring in stuff to change service context
from llama_index import set_global_service_context
from llama_index import ServiceContext

# Bring in embeddings wrapper
from llama_index.embeddings import LangchainEmbedding
# Bring in HF embeddings - need these to represent document chunks
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

locale.getpreferredencoding = lambda: "UTF-8"

app = Flask(__name__)

#==========================================================

# to add Cross Origin Resource Sharing (CORS) policy
from flask_cors import CORS
CORS(app)

#========================================================== 
# Define variable to hold llama2 weights naming
name = "meta-llama/Llama-2-7b-chat-hf"
# Set auth token variable from hugging face
auth_token = ""


# Create tokenizer
tokenizer = AutoTokenizer.from_pretrained(name,
    cache_dir='./model/', use_auth_token=auth_token)


model = AutoModelForCausalLM.from_pretrained(name,
    cache_dir='./model/', use_auth_token=auth_token, torch_dtype=torch.float16,
    rope_scaling={"type": "dynamic", "factor": 2}, load_in_8bit=True)


# Create a system prompt
system_prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as
helpfully as possible, while being safe. Your answers should not include
any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.<</SYS>>
"""

# Throw together the query wrapper
query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")



# Complete the query prompt
query_wrapper_prompt.format(query_str='Asslam-u-Alikum! Here is your answer')



# Create a HF LLM using the llama index wrapper
llm = HuggingFaceLLM(context_window=4096,
                    max_new_tokens=2000, # float('inf') use for infinity,
                    system_prompt=system_prompt,
                    query_wrapper_prompt=query_wrapper_prompt,
                    model=model,
                    tokenizer=tokenizer)


# Create and dl embeddings instance
# !pip install sentence_transformers
embeddings=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

# Create new service context instance
service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embeddings
)
# And set the service context
set_global_service_context(service_context)


# =======================================
# Api Routes Start From here
# =======================================
@app.route('/', methods=["GET","POST"])
def index():
    return "Hello World!"

# @app.route('/api/chatbot')
@app.route('/api/chatbot', methods=["POST"])
def chatBot():

    prompt_result = (request.form['prompt'])

    print(prompt_result)

    
    # Define your question
    question = f"{prompt_result}"

    # Use the LLM predictor to generate a response
    response = service_context.llm_predictor.predict(question)


    # print(output_text)
    return {
        "data" : response,
            "status" : 200,
            "message" : "Response Generate Successfully!"}


if __name__ == "__main__":
    
    # for local run 
    # app.run(debug=False)
    # for live run
    app.run(host='0.0.0.0', port=8000, debug=False)
    
# Islamic GPT

### Set the Anaconda environment

**Download anaconda from this site for window:**
    https://www.anaconda.com/download

After downloading the Anaconda and just install it simply into the window:
    ***create a new environment in anaconda with the name of IslamicGpt***


After setup the anaconda enviroment we need to open anaconda terminal or cmd and run this command there which is given below:
    
    conda activate IslamicGpt

Through out downloading the libraries in series:

Install pytorch

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 --upgrade

Install langchain scipy

    pip install langchain einops accelerate transformers bitsandbytes scipy

Install senencepiece of sentence transformer

    pip install xformers sentencepiece

Install llama-index and lamma_hub for hitting the lamma models

    pip install llama-index==0.7.21 llama_hub==0.0.19

Install accelerate for our CPU Performance

    pip install accelerate

install bitsandbytes for run our model in how many bit like 4 bit, 8 bit, 32 bit etc

    pip install -i https://test.pypi.org/simple/ bitsandbytes

Install pydantic for GPU dedication

    pip install pydantic==1.10.9

### After all of this set finally run this command:

    python index.py


**Note:** Once the require library install in the environment we don't require to install it again and again
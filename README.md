# LLM_QnA_Chatbot
custom data AI chatbot on healthcare dataset

# Create a virtual environment (recommended)
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Root path 
uvicorn main:app --reload

# root/streamlit/ 
    - streamlit run app.py



# Just for knowledge - 
    - utils/train - IN train.py, i mentioned how to finetune model, due to less epoch and small data I'm Not using finetuned model for production 
    - utils/train - In infrense.py, i mentioned how to use finetuned model. 


# work flow - 
    - input csv file must have question and answer column names 
    - once input csv is loaded its data will be stored in faiss index and embedding format
    - next time cheker fucntion will check if file is already loaded once with the same name of file, loader will skip to load and store it again. 
    - Now agent will correct the query and make gramatically correct. 
    - greet Agent - response the greeting queries 
    - simiarity function - responsible to provide answer for given query 
    - if question not match atleast 65% politly decline the answer and custom msg will printed. 
    
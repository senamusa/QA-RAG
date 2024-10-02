from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer

from llama_index.core import SimpleDirectoryReader, Settings, SummaryIndex
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


app = FastAPI()
hf_token = "hf_"

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    token=hf_token,
)

stopping_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

llm = HuggingFaceLLM(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    model_kwargs={
        "token": hf_token,
    },
    generate_kwargs={
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
    },
    tokenizer_name="meta-llama/Llama-3.2-1B-Instruct",
    tokenizer_kwargs={"token": hf_token},
    stopping_ids=stopping_ids,
)

# bge embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model

# Llama-3-8B-Instruct model
Settings.llm = llm

spotify_documents = SimpleDirectoryReader(
    # use smaller data due to memory limitation
    input_files=["spotify_reviews_lite.csv"]
).load_data()

index = SummaryIndex.from_documents(spotify_documents)
query_engine = index.as_query_engine(response_mode="tree_summarize")

def rag_qa(input_str: str) -> str:
    return query_engine.query(f"Write a short answer to the question. Q: {user_query}")

# Define a request model
class UserQuery(BaseModel):
    query: str

# endpoint to get rag answer
@app.post("/user_query")
async def user_query(user_query: UserQuery):
    result = rag_qa(user_query.query)
    return {"result": result}
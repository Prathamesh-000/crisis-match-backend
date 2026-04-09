from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import json

load_dotenv()

app = FastAPI()

# Load LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# 🔥 memory store per user
memory_store = {}

def get_memory(user_id):
    if user_id not in memory_store:
        memory_store[user_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    return memory_store[user_id]

# Request format
class Input(BaseModel):
    message: str
    user_id: str

# Prompt
prompt = PromptTemplate(
    input_variables=["message", "chat_history"],
    template="""
You are an intelligent emergency assistant.

Chat history:
{chat_history}

User message:
{message}

Tasks:
1. Extract intent
2. Extract occupation if mentioned
3. If not mentioned, infer carefully
4. If unclear, set is_occupation_provided = false

Return JSON ONLY:
{
 "intent": "",
 "occupation": "",
 "is_occupation_provided": true/false,
 "is_valid_request": true
}
"""
)

@app.post("/agent")
def run_agent(input: Input):
    memory = get_memory(input.user_id)

    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory
    )

    result = chain.run(message=input.message)

    # ensure JSON
    try:
        parsed = json.loads(result)
    except:
        parsed = {
            "intent": "emergency",
            "occupation": "",
            "is_occupation_provided": False,
            "is_valid_request": True
        }

    return {"output": parsed}
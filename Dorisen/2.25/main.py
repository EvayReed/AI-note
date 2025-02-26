import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
prompt = ChatPromptTemplate.from_template(
    "用三个emoji描述：{item}"
)
model = ChatOpenAI(openai_api_key=api_key)
chain = prompt | model

response = chain.invoke({"item": "程序员的工作日常"})
print(response.content)
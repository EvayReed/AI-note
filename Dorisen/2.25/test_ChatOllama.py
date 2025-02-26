from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(
    model="deepseek-r1:1.5b",
    temperature=0.7,
    base_url="http://localhost:11434",
    stream_usage=True  # 流模式需开启统计
)

response = llm.invoke([
    HumanMessage(content="望梅止渴是什么意思")
])

# 输出内容和token统计
print(f"响应内容: {response.content}")
print(f"输入Token: {response.usage_metadata['input_tokens']}")
print(f"输出Token: {response.usage_metadata['output_tokens']}")
print(f"总消耗Token: {response.usage_metadata['total_tokens']}")
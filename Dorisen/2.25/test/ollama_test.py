from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# 初始化模型
# ollama = ChatOllama(
#     base_url="http://localhost:11434",  # deepseek 服务地址
#     model="deepseek-r1:1.5b"  # 模型名称（需提前下载）
# )

ollama = ChatOllama(
    base_url="http://10.1.4.136:11434/",  # deepseek 服务地址
    model="deepseek-r1:32b"  # 模型名称（需提前下载）
)


# 调用模型生成回复
response = ollama.invoke([
    HumanMessage(content="你是什么模型")
])

print(response.content)
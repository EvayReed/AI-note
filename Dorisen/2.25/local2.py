from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama, OllamaLLM
OllamaChat = ChatOllama(
    model="deepseek-r1:1.5b",
    temperature=0.7,
    base_url="http://localhost:11434"
)

Ollama_LLM = OllamaLLM(
    model="deepseek-r1:1.5b",
    temperature=0.7,
    base_url="http://localhost:11434"
)

print(OllamaChat.invoke([
    AIMessage(role="system", content="你好，我是丁凯乐"),
    HumanMessage(role="user", content="你好，我是凯南"),
    AIMessage(role="system", content="很高兴认识你"),
    HumanMessage(role="system", content="你知道我叫什么吗？")

]))
print(Ollama_LLM.invoke("你好啊，你叫什么名字？"))
# content='<think>\n\n</think>\n\n您好！我是由中国的深度求索（DeepSeek）公司开发的智能助手DeepSeek-R1。如您有任何任何问题，我会尽我所能为您提供帮助。' addiepseek-r1:1.5b', 'created_at': '2025-02-25T09:00:46.11562Z', 'done': True, 'done_reason': 'stop', 'total_duration': 1582564167, 'load_duration': 29937084, 'prompt_eval_count': 28, 'prompt_eval_duration': 288000000, 'eval_count': 40, 'eval_duration': 1041000000, 'message': Message(role='assistant', content='', images=None, tool_calls=None)} id='run-663b5387-d5e5-4ded-8695-30042adfaab6-0' usage_metadata={'input_tokens': 28, 'output_tokens': 40, 'total_tokens': 68}
# <think>
# 
# </think>
#
# 您好！我叫DeepSeek-R1，是一个由深度求索公司开发的智能助手，我会尽我所能为您提供帮助。
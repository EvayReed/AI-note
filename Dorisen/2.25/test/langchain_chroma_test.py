from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
#`BeautifulSoup'解析网页内容：按照标签、类名、ID 等方式来定位和提取你需要的内容
import bs4
#Load HTML pages using `urllib` and parse them with `BeautifulSoup'
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
#文本分割
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 向量嵌入 ::: conda install onnxruntime -c conda-forge
from langchain_community.vectorstores import Chroma
# 有许多嵌入模型
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain_ollama import ChatOllama

# 初始化llm, 让其流式输出
# llm = Ollama(model="deepseek-r1:1.5b",
#              temperature=0.1,
#              top_p=0.4,
#              callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
#              )

# llm = ChatOllama(
#     base_url="http://10.1.4.136:11434/",  # deepseek 服务地址
#     model="deepseek-r1:32b"  # 模型名称（需提前下载）
# )


# ollama = ChatOllama(
#     base_url="http://localhost:11434",  # deepseek 服务地址
#     model="deepseek-r1:1.5b",  # 模型名称（需提前下载）
#     temperature=0.1,
#     top_p=0.4,
#     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
# )

# 模型文件的 URL
# model_url = "http://10.1.4.136:11434/deepseek-r1:32b"

llm = Ollama(base_url="http://10.1.4.136:11434/",  # deepseek 服务地址
             model="deepseek-r1:32b",# 模型名称（需提前下载）
             temperature=0.1,
             top_p=0.4,
             callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
             )

# loader = WebBaseLoader(
#     web_paths=("https://vuejs.org/guide/introduction.html#html",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("content",),
#             # id=("article-root",)
#         )
#     ),
# )

# 加载本地文件

# loader = TextLoader("data_movie/merged_dataset.csv", encoding="utf-8")
#
# docs = loader.load()
# # chunk_overlap：分块的重叠部分
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#
# splits = text_splitter.split_documents(docs)
#
# vectorstore = Chroma.from_documents(documents=splits,
#                                     embedding=OllamaEmbeddings(model="nomic-embed-text"))

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vectorstore = Chroma(
    collection_name="db_movie",
    embedding_function=embeddings,
    persist_directory="./chroma_data",  # 确保与之前存储时使用的目录相同
)

prompt = PromptTemplate(
    input_variables=['context', 'question'],
    template=
    """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the 
    question. you don't know the answer, just say you don't know 
    without any explanation Question: {question} Context: {context} Answer:""",
)
# 向量数据库检索器
retriever = vectorstore.as_retriever()

# qa_chain = RetrievalQA.from_chain_type(
#     llm,
#     retriever=retriever,
#     chain_type_kwargs={"prompt": prompt}
# )
#
# question = "what is Avatar?"
# result = qa_chain.invoke({"query": question})

# 初始化对话历史
conversation_history = []

# 创建一个 RetrievalQA 实例
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

# 连续对话的循环
while True:
    # 从控制台获取用户输入
    user_input = input("You: ")

    # 如果用户输入了退出命令，跳出循环
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break

    # 将对话历史与当前用户输入拼接在一起
    conversation_input = " ".join(conversation_history) + " " + user_input

    # 获取模型的回答
    result = qa_chain.invoke({"query": conversation_input})

    # 打印模型的回答
    print(f"Bot: {result['result']}")

    # 更新对话历史
    conversation_history.append(f"User: {user_input}")
    conversation_history.append(f"Bot: {result['result']}")
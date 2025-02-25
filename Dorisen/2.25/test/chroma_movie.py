from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
#https://python.langchain.com/docs/integrations/vectorstores/chroma/#basic-initialization
from langchain_chroma import Chroma
import chardet
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import pandas as pd


# # 读取原始文档
# raw_documents_movie = TextLoader('data_movie/merged_dataset.csv', encoding='utf-8').load()
#
# # 分割文档
# text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
# documents_movie = text_splitter.split_documents(raw_documents_movie)
# print("documents nums:", documents_movie.__len__())

# 读取 CSV 文件
df = pd.read_csv('data_movie/merged_dataset.csv')

# 假设每行代表一个电影的记录，你可以选择把这些记录的具体内容提取为文本
# 例如，将每一行的电影名称、年份、类型等字段拼接为一个长文本来处理
raw_documents_movie = df.apply(lambda row: f"Name: {row['name']}, Year: {row['year']}, Rating: {row['rating']}, Genres: {row['genres']}, Run Length: {row['run_length']}, Release Date: {row['release_date']}, Num Raters: {row['num_raters']}, Num Reviews: {row['num_reviews']}", axis=1).tolist()

# 创建文本分割器，设置每个块的最大长度为300
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)

# 将每个文本转换为 Document 对象，并分割每一行文本
documents_movie = []
for doc in raw_documents_movie:
    document = Document(page_content=doc)  # 创建 Document 对象
    documents_movie.extend(text_splitter.split_documents([document]))  # 分割文档

# 生成向量（embedding）
# model_id = "damo/nlp_corom_sentence-embedding_chinese-base"
# embeddings = ModelScopeEmbeddings(model_id=model_id)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# db = Chroma.from_documents(documents_movie, embedding=embeddings)

db = Chroma(
    collection_name="db_movie",
    embedding_function=embeddings,
    persist_directory="./chroma_data",  # Where to save data locally, remove if not necessary
)
db.add_documents(documents=documents_movie)


# 检索
query = "what is Avatar?？"
docs = db.similarity_search(query, k=1)

# 打印结果
for doc in docs:
    print("===")
    print(":", doc.metadata)
    print("page_content:", doc.page_content)

import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

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

print("documents nums:", len(documents_movie))
import chromadb
# setup Chroma in-memory, for easy prototyping. Can add persistence easily!
#client = chromadb.Client()
#持久化
client = chromadb.PersistentClient(path="chroma_data")


# Create collection. get_collection, get_or_create_collection, delete_collection also available!
collection = client.create_collection("all-my-documents")

# 加载本地文档
with open("data_movie/merged_dataset.csv", "r") as file:
    document = file.read()

# 添加文档到集合
collection.add(documents=[document], metadatas=[{"source": "local"}], ids=["doclocal"])

# Add docs to the collection. Can also update and delete. Row-based API coming soon!
collection.add(
    documents=["This is document1", "This is document2"], # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
    metadatas=[{"source": "notion"}, {"source": "google-docs"}], # filter on these!
    ids=["doc1", "doc2"], # unique for each doc
)

# Query/search 2 most similar results. You can also .get by id
results = collection.query(
    query_texts=["This is a query document"],
    n_results=2,
    # where={"metadata_field": "is_equal_to_this"}, # optional filter
    # where_document={"$contains":"search_string"}  # optional filter
)
# returns a nanosecond heartbeat. Useful for making sure the client remains connected.
#client.heartbeat()
# Empties and completely resets the database. ⚠️ This is destructive and not reversible.
#client.reset()

print(results)

# collection = client.get_collection(name="test") # Get a collection object from an existing collection, by name. Will raise an exception if it's not found.
# collection = client.get_or_create_collection(name="test") # Get a collection object from an existing collection, by name. If it doesn't exist, create it.
# client.delete_collection(name="my_collection") # Delete a collection and all associated embeddings, documents, and metadata. ⚠️ This is destructive and not reversible
# collection.peek() # returns a list of the first 10 items in the collection
# collection.count() # returns the number of items in the collection
# collection.modify(name="new_name") # Rename the collection
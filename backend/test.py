# from tasks.create_vectordatabase import CreateVectordatabase
# c = CreateVectordatabase()
# c.post()


from langchain_chroma import Chroma

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model_name='BAAI/bge-m3',model_kwargs={'device': 'cpu'})
vectorstore = Chroma(persist_directory="../frontend/chroma", embedding_function=embedding).as_retriever(
search_type = "similarity",
search_kwargs = {"k":10}
)
print(vectorstore.invoke("Hustenbonbon"))
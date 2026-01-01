from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_function():
    #print("Loading Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    return embeddings

if __name__ == "__main__":
    get_embedding_function()

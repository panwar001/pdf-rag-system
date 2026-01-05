pdf-rag-system

RAG System for PDF Q&A

* pip install requirements.txt

* Embedding Model - HuggingFaceEmbeddings [all-mpnet-base-v2]

* Vector DB - chromadb

* ollama tool to pull llama3 model

* Run Chatbot using command - streamlit run chatbot.py

RAG pipeline - 

✅ Upload / read a PDF

✅ Extract text

✅ Chunk intelligently

✅ Generate embeddings

✅ Store them in ChromaDB

✅ Uses collection name = documents_collection

✅ Ready for RAG, LangChain, FastAPI, Ollama/OpenAI later

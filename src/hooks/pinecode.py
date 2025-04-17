import os
from dotenv import load_dotenv
from typing import List
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema.document import Document
from pinecone import Pinecone, ServerlessSpec
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Setup Embeddings & LLM
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# Initialize Pinecone (new style)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "demo"
# if pc.has_index(index_name):
#     print(f"Deleting index: {index_name}")
#     pc.delete_index(index_name)


if not pc.has_index(name=index_name):
    # Create a new index
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

pc.describe_index(name=index_name)
index = pc.Index(index_name)
vectorstore = PineconeVectorStore(
    index_name=index_name, embedding=embedding_model, text_key="text"
)

embedding = embedding_model.embed_query("test")


def ingest_data(data: List[str]) -> bool:
    try:
        docs = [Document(page_content=chunk) for chunk in data]
        vectorstore.add_documents(docs)
        # index.upsert(vectors=docs)
        return True
    except Exception as e:
        print("❌ Error ingesting data:", e)
        return False


def Query_pinecone(query: str, top_k: int = 1) -> List[str]:
    try:
        # Perform similarity search in Pinecone vectorstore
        results = vectorstore.similarity_search(query, k=top_k)

        # Initialize list to hold results
        vector_results = [r.page_content for r in results]

        # Check if results are empty, and return an empty list if so
        if not vector_results:
            return []

        # Otherwise, return the content of the search results
        content = [result for result in vector_results]
        return content

    except Exception as e:
        print("❌ Error querying Pinecone:", e)
        return []

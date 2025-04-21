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
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# Initialize Pinecone (new style)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "demo"
deleted = False
# if pc.has_index(index_name) and not deleted:
#     print(f"Deleting index: {index_name}")
#     pc.delete_index(index_name)
#     deleted = True


# if not pc.has_index(name=index_name):
#     # Create a new index
#     pc.create_index(
#         name=index_name,
#         dimension=768,
#         metric="dotproduct",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1"),
#     )

pc.describe_index(name=index_name)
index = pc.Index(index_name)
vectorstore = PineconeVectorStore(
    index_name=index_name, embedding=embedding_model, text_key="text"
)

# embedding = embedding_model.embed_query("test")


def ingest_data(data: List[dict]) -> bool:
    try:
        print()
        if not data or len(data) == 0:
            print("❌ No data provided for ingestion.")
            return False
        docs = [
            Document(page_content=item["text"], metadata={"url": item["url"], "title": item.get("title", "Untitled Paper")})
            for item in data
        ]

        vectorstore.add_documents(docs)
        print(f"✅ Ingested {len(docs)} documents.")
        return True
    except Exception as e:
        print("❌ Error ingesting data:", e)
        return False



def Query_pinecone(query: str, top_k: int = 1) -> List[dict]:
    try:
        print(f"🔍 Querying Pinecone with: {query}")
        # Perform semantic similarity search
        results = vectorstore.similarity_search(query, k=top_k)

        # Ensure results are returned in the correct format
        return [
            {
                "text": doc.page_content,
                "url": doc.metadata.get("url", "unknown"),
                "title": doc.metadata.get("title", "unknown"),
            }
            for doc in results
        ]

    except Exception as e:
        print(f"❌ Error querying Pinecone: {e}")
        return []

def Retriver(query: str, top_k: int = 5) -> List[dict]:
    try:
        print(f"🔍 Retrieving documents for query: {query}")
        results = vectorstore.as_retriever(search_kwargs={"k": top_k})

        return [
            {
                "text": doc.page_content,
                "url": doc.metadata.get("url", "unknown"), 
                "title": doc.metadata.get("title", "unknown"),
            }
            for doc in results
        ]

    except Exception as e:
        print(f"❌ Error retrieving documents from Pinecone: {e}")
        return []

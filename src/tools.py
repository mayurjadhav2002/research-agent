from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
from src.hooks.chunks import extract_elements
import urllib.request
import xml.etree.ElementTree as ET
from src.hooks.download import Download_PDF
import io
from src.db import init_db, paper_exists, save_paper
# from PyPDF2 import PdfReader
from src.hooks.pinecode import Query_pinecone, ingest_data, Retriver
import os
UPLOAD_FOLDER="uploads"

def search_arvix_website(query: str, max_results: int = 5, type="application") -> list[dict]:
    init_db()

    url = f"http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(str(query))}&start=0&max_results={max_results}"
    print(f"🔍 Searching arXiv for {query}...\n{url}")
    data = urllib.request.urlopen(url)
    res = data.read().decode("utf-8")

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(res)

    chunks = []
    processed_papers = [] 

    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns).text.strip()
        pdf_url = entry.find("atom:id", ns).text.replace("/abs/", "/pdf/")

        if paper_exists(pdf_url):
            print(f"🟡 Skipping already-ingested paper: {title}")
            continue

        print(f"⬇️ Downloading new paper: {title}")
        localpath = Download_PDF(pdf_url)
        result = extract_elements(localpath)
        paper_chunks = [
            {
                "text": text.text.strip(),
                "url": pdf_url,
                "title": title
            }
            for text in result["texts"]
            if hasattr(text, "text")
        ]
        

        chunks.extend(paper_chunks)
        Download_PDF(pdf_url, remove=True)
        save_paper(pdf_url, title)

        processed_papers.append({"title": title, "url": pdf_url})  
    
    
    print(f"\n🗂️ Scanning local upload folder: {UPLOAD_FOLDER}")
    for filename in os.listdir(UPLOAD_FOLDER):
        if not filename.lower().endswith(".pdf") or filename.endswith("_scraped.pdf"):
            continue  # Skip non-PDFs and already scraped ones

        full_path = os.path.join(UPLOAD_FOLDER, filename)
        file_id = f"local:{filename}"

        if paper_exists(file_id):
            print(f"🟡 Skipping already-processed local file (in DB): {filename}")
            continue

        print(f"📥 Ingesting uploaded file: {filename}")
        result = extract_elements(full_path)

        file_chunks = [
            {"text": text.text.strip(), "url": file_id, "title": filename}
            for text in result["texts"]
            if hasattr(text, "text")
        ]

        chunks.extend(file_chunks)
        save_paper(file_id, filename)

        new_filename = filename.rsplit(".", 1)[0] + "_scraped.pdf"
        new_path = os.path.join(UPLOAD_FOLDER, new_filename)
        processed_papers.append({"title": filename, "url": f"/uploads/{new_filename}"})
        if not os.path.exists(new_path):
            os.rename(full_path, new_path)
            print(f"✅ Renamed {filename} -> {new_filename}")

    if chunks:
        print(f"📚 Found {len(chunks)} chunks. Ingesting...")
        ingest_data(chunks)
    else:
        print("✅ All papers already processed.")

    return processed_papers 

    
    
def Query_Existing_Papers(query: str) -> list[str]:
    try:
        results = Retriver(query)
        if not results:
            print(f"⚠️ No matching papers found in Pinecone for query: {query}")
            return [] 
        else:
            print("🔗 Found papers in Pinecone:", results)
            return results

    except Exception as e:
        print(f"❌ Error in Query_Existing_Papers: {e}")
        return []





search = DuckDuckGoSearchRun()


wikipedia = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)


def save_file_to_txt(data: str, filename: str = "output.txt") -> str:
    print(f"💾 Saving data to {filename}...")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    formatted_text = f"-----\n{timestamp}\n---\n{data}\n---\n"
    with open(filename, "a", encoding="utf-8") as file:
        file.write(formatted_text)
    return f"File saved successfully to {filename}!"


wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia)


save_tool = Tool(
    name="save_to_txt",
    func=save_file_to_txt,
    description="saves structured research data to a text file.",
)

research_tool = Tool(
    name="research_tool",
    func=search_arvix_website,
    description="If you dont find answer in pinecone, Search the arvix website for new research papers. generate new query and search. Useful for when you need to find information that is not in the Existing Pinecone database, or Duck duckgo or Wikipedia database.",
)

search_in_papers = Tool(
    name="Search_in_papers",
    func=Query_Existing_Papers,
    description=(
           "Internal Paper search tool. Primary tool for answering technical and academic queries using research papers stored in Pinecone. "
        "Returns results with title and URL. Always include these papers in the 'research_papers' field of the final response."
     ),)

search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information. Useful for when you need to find information that is not in the Wikipedia database.",
)

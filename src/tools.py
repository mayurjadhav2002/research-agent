from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
from src.hooks.chunks import extract_elements
import urllib.request
import xml.etree.ElementTree as ET
from src.hooks.download import Download_PDF
import io

# from PyPDF2 import PdfReader
from src.hooks.pinecode import Query_pinecone, ingest_data


def search_arvix_website(query: str, max_results: int = 1) -> list[str]:
    url = f"http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query)}&start=0&max_results={max_results}"
    print(f"🔍 Searching arvix website for {query}...\n{url}")
    data = urllib.request.urlopen(url)
    res = data.read().decode("utf-8")

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(res)

    papers = [
        entry.find("atom:id", ns).text.replace("/abs/", "/pdf/")
        for entry in root.findall("atom:entry", ns)
    ]
    if not papers:
        print("No papers found.")
        return None
    chunks = []
    for paper in papers:
        localpath = Download_PDF(paper)
        result = extract_elements(localpath)
        chunks.extend(
            [text.text.strip() for text in result["texts"] if hasattr(text, "text")]
        )
        Download_PDF(paper, remove=True)

    print("found papers, loading it to memory...")
    ingest_data(chunks)
    return chunks


def Query_Existing_Papers(query: str) -> list[str]:
    try:
        response = Query_pinecone(query)
        return response
    except Exception as e:
        print("Error querying Pinecone: ", e)
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
    name="research",
    func=search_arvix_website,
    description="Search the arvix website for new research papers. Useful for when you need to find information that is not in the Existing Pinecone database, or Duck duckgo or Wikipedia database.",
)

Search_in_papers = Tool(
    name="Search_in_papers",
    func=Query_Existing_Papers,
    description="Search the existing papers in the database for information. Useful for when you need to find information that is not in the Wikipedia database.",
)

search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information. Useful for when you need to find information that is not in the Wikipedia database.",
)

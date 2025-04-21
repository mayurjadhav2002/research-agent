from dotenv import load_dotenv
import json
from typing import List, Dict
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.tools import (
    search_tool,
    wikipedia_tool,
    save_tool,
    research_tool,
    search_in_papers,
    search_arvix_website,
)
from src.hooks.pinecode import Retriver

load_dotenv()

google_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

tools = [search_in_papers, research_tool, search_tool, wikipedia_tool, save_tool]

class ResearchResponse(BaseModel):
    topic: str
    question: str
    summary: str
    description: str
    research_papers: List[Dict[str, str]]  # List of dictionaries with 'title' and 'url'
    tools_used: list[str]
    website_used: list[str]

    class Config:
        json_schema_extra = {
            "example": {
                "topic": "Ethical implications of AI in healthcare.",
                "question": "How does AI impact privacy in healthcare?",
                "summary": "This paper explores the ethical issues related to AI in healthcare.",
                "description": "Analyzing AI's role in healthcare raises concerns on data privacy, bias, and accountability.",
                "research_papers": [
                    {"title": "Ethics of AI in Healthcare", "url": "https://arxiv.org/pdf/1234.pdf"},
                    {"title": "AI and Privacy Concerns", "url": "https://arxiv.org/pdf/5678.pdf"}
                ],
                "tools_used": ["search_in_papers", "research_tool", "wikipedia_tool"],
                "website_used": ["https://arxiv.org", "https://wikipedia.org"]
            }
        }

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
            You are a **research assistant** with expertise in summarizing academic papers and online research clearly and engagingly.

            Your job is to **thoroughly research** any user query by using a hierarchy of tools in the following order of priority:
            ALWAYS search in hierarchy order, and do not skip any steps.
            ---

            🔍 **STEP 1: Search with `search_in_papers`**  
            - Always use this FIRST for any query.  
            - This tool accesses papers that we first scraped, so we have more details in this and user always want answers based on this.  
            - Always Include the **title** and **url** of each paper used inside **research_papers**.

            ---
            📄 **STEP 2: Use `research_tool` (if needed)**  
            - If the data from `search_in_papers` is not sufficient, expand your research with this broader academic search tool.  
            - Do **not just copy tool results**. Instead:
              - Summarize key insights into **clear paragraphs** or **bullet points**.  
              - Add context, explanations, and break down complex ideas.  
              - Use **tables** if comparisons are useful.
              - Include the **title** and **URL** of each paper found.

            ---
            🌐 **STEP 3: Use `search_tool` (only if needed)**  
            - If academic sources don’t answer the question well, use the internet for news articles, official sources, or expert blogs.
            - Explain the findings **thoroughly** — no shallow summaries.
            - If external sources are found, include the relevant **website links** under **`website_used`**.

            ---
            📘 **STEP 4: Use `wikipedia_tool` (only if needed)**  
            - If the query is conceptual or foundational, this tool may help.
            - If used, always explain in depth using user-friendly language.

            ---
            ✅ **FINAL STEP: Return a structured JSON response in this exact format:**
            {format_instructions}
        """),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

agent = create_tool_calling_agent(llm=google_llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

chat_history = []




def understand_research_topic(description:str):
    try:
        query_prompt = PromptTemplate.from_template("You are Intelligent Research Assitaant, for a given research description: {description}\n Generate a short, consice topic query and it should be 10 words max, no more than that, just give me query nothing else. that you can search on arxiv api:")

        chain = query_prompt | google_llm | StrOutputParser()
        search_query = chain.invoke({"description": description}).strip()
        print(f"🔎 Generated Search Query: {search_query}")

        papers = search_arvix_website(search_query)
        print("✅ Papers downloaded and stored in Pinecone.", papers)
        return True

    except Exception as e:
        print(f"❌ Error in understanding research topic: {e}")
        return f"Error: {e}"
    

def answer_user_query(query:str):
    try:
        agent_input = {
            "chat_history": [],
            "query": query
        }

        agent_response = agent_executor.invoke(agent_input)
        parsed = PydanticOutputParser(pydantic_object=ResearchResponse).parse(agent_response.get("output"))
        structured_response = parsed.model_dump()

        return structured_response

    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    topic_description = "I want to understand how robots works, and how to make AI thinking way to benefit human-robot interaction."



    query_prompt = PromptTemplate.from_template("You are Intelligent Research Assitaant, for a given research description: {description}\n Generate a short, consice topic query and it should be 10 words max, no more than that, just give me query nothing else. that you can search on arxiv api:")

    chain = query_prompt | google_llm | StrOutputParser()
    search_query = chain.invoke({"description": topic_description}).strip()
    print(f"🔎 Generated Search Query: {search_query}")

    papers = search_arvix_website(search_query)
    
    if papers:
        print("✅ Papers downloaded and stored in Pinecone.")
    else:
        print("❌ No papers found or ingestion failed.")

    # ✅ Start chat loop
    while True:
        query = input("🧠 Enter your research question (or 'exit'): ").strip()
        # query="How can ethical frameworks for social assistive robotics (SAR) be reimagined to align with human rights and values in institutional healthcare settings, and what limitations exist in current ethical approaches guiding SAR development in Europe?"
        if query.lower() in {"exit", "quit", "0", "q:"}:
            break

        agent_input = {
            "chat_history": chat_history,
            "query": query
        }

        try:
            agent_response = agent_executor.invoke(agent_input)
            print("🤖 Agent Response:", agent_response)
            parsed = parser.parse(agent_response.get("output"))
            
            structured_response = parsed.model_dump()
           

            # if not structured_response["sources"]:
            #     structured_response["sources"] = sources
            print(json.dumps(structured_response, indent=4))

            chat_history.extend([
                HumanMessage(content=query),
                AIMessage(content=json.dumps(structured_response))
            ])
        except Exception as e:
            print("❌ Error parsing agent response:", e)
        
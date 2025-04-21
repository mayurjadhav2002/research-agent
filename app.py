from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
import json
from src.tools import (
    search_tool,
    wikipedia_tool,
    save_tool,
    research_tool,
    Search_in_papers,
    search_arvix_website,
)

load_dotenv()

google_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.9)

tools = [search_tool, wikipedia_tool, save_tool, research_tool, Search_in_papers]


class ResearchResponse(BaseModel):
    topic: str
    question: str
    summary: str
    description: str
    sources: list[str]
    tools_used: list[str]
    website_used: list[str]


parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful research assistant that will generate structured, accurate answers using appropriate tools.
Wrap the output in this format and provide no other text or explanation:
\n{format_instructions}""",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

Agent = create_tool_calling_agent(llm=google_llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=Agent, tools=tools, verbose=True)

if __name__ == "__main__":
    topic_description = "I want to understand how robots think, and how to make AI thinking ethical to benefit human-robot interaction."

    query_prompt = PromptTemplate.from_template(
        "Given the following research project description, generate a natural-language search phrase (one sentence, no special characters):\n\nDescription: {description}\n\nSearch topic:"
    )
    chain = query_prompt | google_llm | StrOutputParser()
    search_query = chain.invoke({"description": topic_description})
    print(f"🔎 Generated Search Query: {search_query.strip()}")

    # Ingest papers
    papers = search_arvix_website(search_query.strip())
    if papers:
        print("✅ Papers downloaded and stored in Pinecone.")
    else:
        print("❌ No papers found or ingestion failed.")

    while True:
        query = input("Enter a query: ")
        if query == "exit" or query == "quit" or query == "q:" or query=="0":
            break
        # query = "What are the core ethical principles that should guide the development and deployment of intelligent autonomous robots across different domains?"

        # Try Pinecone first
        pinecone_result = Search_in_papers.invoke({"query": query})
        if pinecone_result and len(pinecone_result) > 0:
            summary_prompt = PromptTemplate.from_template(
                """Summarize the following research content related to '{query}' in a structured response: \n\n{content}\n\nRespond in this format and use markdown if possible to make it more readable:\n{format_instructions}"""
            ).partial(format_instructions=parser.get_format_instructions())

            summary_chain = summary_prompt | google_llm | parser

            structured = summary_chain.invoke(
                {"query": query, "content": "\n".join(pinecone_result)}
            )
            print(json.dumps(structured.model_dump(), indent=4))

        else:
            print("🔍 No data in Pinecone. Falling back to external tools...")
            agent_response = agent_executor.invoke({"query": query})
        try:
            parsed = parser.parse(agent_response.get("output"))
            # print("📘 Structured Agent Response:\n", parsed)
            print(json.dumps(parsed.model_dump(), indent=4))
        except Exception as e:
            print("❌ Error parsing agent response:", e)

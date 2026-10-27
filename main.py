from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

@tool
def search(query: str) -> str:
    """
    Tool that searches over the internet.

    Args:
        query: The search query.
    Returns:
        The search results.
    """
    print(f"Searching for: {query}")
    return "Paris, France: 14°C, light rain, humidity 82%, wind 15 km/h."

llm = ChatOllama(model="qwen3.5:2b ", temperature=0)
# llm = ChatOpenAI(model_name="gpt-6-luna")
tools = [search]
agent = create_agent(model=llm, tools=tools)

def main():
    result = agent.invoke({"messages": HumanMessage(content="What is the wheather in Paris, France?")})
    print(result)


if __name__ == "__main__":
    main()

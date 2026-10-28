from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

load_dotenv()


# llm = ChatOllama(model="qwen3.5:2b", temperature=0)
llm = ChatOpenAI(model_name="gpt-6-luna")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="Search for 3 job openings for ai eningeering roles using langchain in Madrid and list their details"
            )
        }
    )
    print(result)


if __name__ == "__main__":
    main()

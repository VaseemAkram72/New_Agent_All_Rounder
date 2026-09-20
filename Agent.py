


from dotenv import load_dotenv
import os
import requests

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from tavily import TavilyClient
from rich import print
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call




print("Mistral API key:", bool(os.getenv("MISTRAL_API_KEY")))
print("OpenWeather API key:", bool(os.getenv("OPENWEATHER_API_KEY")))
print("Tavily API key:", bool(os.getenv("TAVILY_API_KEY")))


# WEATHER TOOL

@tool
def get_weather(city: str) -> str:
    """Get current weather of a city."""

    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return "OPENWEATHER_API_KEY is missing."

    url = (
        f"http://api.openweathermap.org/data/2.5/weather"
        f"?q={city},IN&appid={api_key}&units=metric"
    )

    response = requests.get(url, timeout=10)
    data = response.json()

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"Weather in {city}: {desc}, {temp}°C"


# NEWS TOOL


tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


@tool
def get_news(city: str) -> str:
    """Get latest news about a city."""

    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )

    results = response.get("results", [])

    if not results:
        return f"No news found for {city}"

    news_list = []

    for r in results:

        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")

        news_list.append(
            f"- {title}\n"
            f"  🔗 {url}\n"
            f"  📝 {snippet[:100]}..."
        )

    return (
        f"Latest news in {city}:\n\n"
        + "\n\n".join(news_list)
    )


# LLM

llm = ChatMistralAI(
    model="ministral-3b-2512",
    temperature=0
)


# HUMAN APPROVAL

@wrap_tool_call
def human_approval(request, handler):

    tool_name = request.tool_call["name"]

    confirm = input(
        f"Agent wants to call '{tool_name}'. "
        f"Approve? (yes/no): "
    )

    if confirm.lower() != "yes":

        return ToolMessage(
            content="Tool call denied by user.",
            tool_call_id=request.tool_call["id"]
        )

    return handler(request)


# CREATE AGENT

agent = create_agent(
    llm,
    tools=[
        get_weather,
        get_news
    ],
    system_prompt=(
        "You are a helpful city assistant. "
        "You can provide weather and latest city news. "
        "Use tools when necessary."
    ),
    middleware=[
        human_approval
    ]
)


# CHAT LOOP

print("City Agent | type exit to quit")

while True:

    user_input = input("You : ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    try:

        result = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        })

        print("Bot :", result["messages"][-1].content)

    except Exception as e:

        print("\n❌ Error:")
        print(e)
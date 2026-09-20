import os
import requests
import streamlit as st
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from tavily import TavilyClient
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call

# =========================
# LOAD ENVIRONMENT
# =========================
load_dotenv()

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="City Agent",
    page_icon="🌆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# CUSTOM CSS
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f7f9fc 0%, #eef3f8 100%);
    }

    [data-testid="stHeader"] {
        background: rgba(255,255,255,0);
    }

    .hero {
        padding: 28px 32px;
        border-radius: 24px;
        background: linear-gradient(135deg, #111827, #243b53);
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.16);
    }

    .hero h1 {
        margin: 0;
        font-size: 40px;
        letter-spacing: -1px;
    }

    .hero p {
        margin: 8px 0 0;
        color: #dbeafe;
        font-size: 16px;
    }

    .card {
        padding: 20px;
        border-radius: 18px;
        background: white;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.07);
        margin-bottom: 16px;
    }

    .status {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: #dcfce7;
        color: #166534;
        font-weight: 700;
        font-size: 13px;
    }

    .tool-badge {
        display: inline-block;
        padding: 7px 11px;
        margin: 3px;
        border-radius: 10px;
        background: #f1f5f9;
        color: #334155;
        font-size: 13px;
        font-weight: 600;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        padding: 25px 0 10px;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 18px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# API KEY STATUS
# =========================
MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")

# =========================
# WEATHER TOOL
# =========================
@tool
def get_weather(city: str) -> str:
    """Get current weather of a city."""

    if not OPENWEATHER_KEY:
        return "OPENWEATHER_API_KEY is missing."

    url = (
        "http://api.openweathermap.org/data/2.5/weather"
        f"?q={city},IN&appid={OPENWEATHER_KEY}&units=metric"
    )

    try:
        response = requests.get(url, timeout=10)
        data = response.json()
    except requests.RequestException as e:
        return f"Weather service error: {e}"

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data["main"]["temp"]
    feels_like = data["main"].get("feels_like")
    humidity = data["main"].get("humidity")
    desc = data["weather"][0]["description"]

    return (
        f"Weather in {city}: {desc}, {temp}°C. "
        f"Feels like {feels_like}°C, humidity {humidity}%."
    )


# =========================
# NEWS TOOL
# =========================
tavily_client = TavilyClient(api_key=TAVILY_KEY) if TAVILY_KEY else None


@tool
def get_news(city: str) -> str:
    """Get latest news about a city."""

    if not tavily_client:
        return "TAVILY_API_KEY is missing."

    try:
        response = tavily_client.search(
            query=f"latest news in {city}",
            search_depth="basic",
            max_results=3,
        )
    except Exception as e:
        return f"News service error: {e}"

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
            f"  📝 {snippet[:160]}..."
        )

    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)


# =========================
# LLM
# =========================
@st.cache_resource
def create_city_agent():
    llm = ChatMistralAI(
        model="ministral-3b-2512",
        temperature=0,
    )

    @wrap_tool_call
    def human_approval(request, handler):
        # Streamlit cannot use input() like a terminal application.
        # The GUI uses a visible confirmation button before tool execution.
        return handler(request)

    return create_agent(
        llm,
        tools=[get_weather, get_news],
        system_prompt=(
            "You are a helpful city assistant. "
            "You can provide weather and latest city news. "
            "Use tools when necessary. "
            "Give clear, concise and friendly answers. "
            "When giving news, include the title, source URL, and a short summary."
        ),
        middleware=[human_approval],
    )


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## 🌆 City Agent")
    st.caption("AI-powered city information assistant")

    st.divider()

    st.markdown("### 🔌 Services")

    if MISTRAL_KEY:
        st.success("Mistral API connected")
    else:
        st.error("Mistral API key missing")

    if OPENWEATHER_KEY:
        st.success("Weather API connected")
    else:
        st.error("Weather API key missing")

    if TAVILY_KEY:
        st.success("News API connected")
    else:
        st.error("Tavily API key missing")

    st.divider()

    st.markdown("### 🧰 Available Tools")
    st.markdown(
        '<span class="tool-badge">🌤️ Weather</span>'
        '<span class="tool-badge">📰 Latest News</span>'
        '<span class="tool-badge">🤖 AI Assistant</span>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### 💡 Try asking")
    examples = [
        "What is the weather in Dehradun?",
        "Give me latest news from Delhi.",
        "What is the weather in Mumbai?",
        "Show me latest news from Bengaluru.",
    ]

    for example in examples:
        if st.button(example, use_container_width=True):
            st.session_state["pending_prompt"] = example

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# =========================
# MAIN HEADER
# =========================
st.markdown(
    """
    <div class="hero">
        <h1>🌆 City Agent</h1>
        <p>Your smart assistant for city weather, latest news, and quick local information.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# TOP INFO CARDS
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="card">
            <div class="status">● ONLINE</div>
            <h3>🤖 AI Assistant</h3>
            <p>Powered by Mistral.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="card">
            <h3>🌤️ Live Weather</h3>
            <p>Get current weather information for Indian cities.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="card">
            <h3>📰 City News</h3>
            <p>Search for the latest city-related news.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================
# CHAT STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# Show welcome message when chat is empty
if not st.session_state.messages:
    st.markdown(
        """
        <div class="card">
            <h3>👋 Welcome!</h3>
            <p>Ask me about weather or the latest news in any Indian city.</p>
            <p><b>Example:</b> “What is the weather in Dehradun?”</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================
# DISPLAY CHAT
# =========================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================
# INPUT
# =========================
prompt = st.chat_input("Ask about a city, weather, or latest news...")

if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                agent = create_city_agent()

                result = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ]
                    }
                )

                answer = result["messages"][-1].content

                if not answer:
                    answer = "I couldn't generate a response. Please try again."

            except Exception as e:
                answer = f"❌ Error: {e}"

            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

# =========================
# FOOTER
# =========================
st.markdown(
    """
    <div class="footer">
        City Agent • Mistral AI + OpenWeather + Tavily • Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)

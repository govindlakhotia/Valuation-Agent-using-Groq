import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from textwrap import dedent
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.yfinance import YFinanceTools
import os

# Set API key securely (on Streamlit Cloud, use secrets instead)
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Define the agent
Technical_analysis_agent = Agent(
    model=Groq(id="llama3-70b-8192"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            historical_prices=True,
            company_info=True,
            company_news=True,
        )
    ],
    instructions=dedent("""\
        You are a skilled Technical Analyst specializing in chart patterns, indicators, and trend analysis! 📈
        Perform a full technical and valuation-based analysis using Yahoo Finance data.
    """)
)

# Function to plot chart
def plot_stock_chart(ticker):
    data = yf.Ticker(ticker).history(period="6mo")
    if data.empty:
        st.warning("No data found for this ticker.")
        return
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Close'])
    ax.set_title(f"{ticker} - Closing Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True)
    st.pyplot(fig)

# Streamlit UI
st.set_page_config(page_title="Valuation Agent Dashboard")
st.title("📈 Valuation Agent Dashboard")

ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL, TSLA):")

if ticker:
    with st.spinner("Running technical analysis..."):
        prompt = f"Do a technical analysis of {ticker} and let me know if it is undervalued or overvalued?"
        response = Technical_analysis_agent.run(prompt)
        if hasattr(response, "content"):
            analysis = response.content.replace("\\n", "\n").replace('\\"', '"')
            st.markdown(analysis, unsafe_allow_html=True)
        else:
            st.error("No valid response from the agent.")

    st.subheader("📉 Stock Price Chart")
    plot_stock_chart(ticker)

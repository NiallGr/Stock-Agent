import asyncio
import json
import os
import yfinance as yf
from agents import Agent, Runner

def get_fast_info(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    fi = dict(getattr(t, "fast_info", {}) or {})
    return {"ticker": ticker, **fi} 

async def main():
    # Load API key
    with open("config.json") as f:
        cfg = json.load(f)
    os.environ["OPENAI_API_KEY"] = cfg["OPENAI_API_KEY"]

    ticker = input("Enter ticker (e.g., AAPL, MSFT, TSLA): ").strip().upper()
    if not ticker:
        print("No ticker provided. Exiting.")
        return

    data = get_fast_info(ticker)

    agent = Agent(
        name="StocksAssistant",
        instructions=(
            "You are a concise equities assistant.\n"
            "• ONLY use the provided JSON snapshot from fast_info.\n"
            "• Summarise in 3–5 short bullets: price & day move, 52-week range, "
            "market cap, plain-English momentum.\n"
            "• End with: 'Not financial advice'."
        ),
        model="gpt-4.1-mini",
    )

    prompt = f"DATA:\n{json.dumps(data, ensure_ascii=False)}"
    result = await Runner.run(agent, prompt)

    print("\n=== Agent Output ===")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
import asyncio
from modules.connector import initialize, shutdown, get_api

async def check_price():
    if not await initialize():
        print("Failed to connect")
        return
    api = get_api()
    resp = await api.send({"ticks": "R_100"})
    print(f"Current Tick R_100: {resp.get('tick', {}).get('quote')}")
    await shutdown()

if __name__ == "__main__":
    asyncio.run(check_price())

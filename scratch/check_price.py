import asyncio
import json
import websockets
import os
from dotenv import load_dotenv

load_dotenv()

async def get_price():
    uri = f"wss://ws.binaryws.com/websockets/v3?app_id=1089"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"authorize": os.getenv("DERIV_API_TOKEN")}))
        await ws.recv()
        
        await ws.send(json.dumps({"ticks": "R_75", "subscribe": 0}))
        resp = await ws.recv()
        data = json.loads(resp)
        print(f"Current R_75 tick: {json.dumps(data, indent=2)}")

if __name__ == "__main__":
    asyncio.run(get_price())

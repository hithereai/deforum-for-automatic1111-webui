import pickle
import asyncio
import websockets
from .args import RealTimeControlArgs

async def sendAsync(value):
    async with websockets.connect("ws://localhost:8765") as websocket:
        await websocket.send(pickle.dumps(value))
        message = await websocket.recv()
        return message
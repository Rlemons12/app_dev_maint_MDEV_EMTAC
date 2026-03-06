from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import logging
import asyncio
import json
logging.basicConfig(level=logging.INFO)
import time
app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/sse")
async def sse():

    async def stream():

        logging.info("PyCharm connected to MCP SSE stream")

        while True:
            await asyncio.sleep(20)
            yield "event: ping\ndata: {}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/mcp")
async def mcp_chat(request: Request):

    payload = await request.json()

    logging.info("MCP request received")
    logging.info(payload)

    return {
        "content": "Stub AI response from EMTAC MCP server."
    }
@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "emtac-local",
                "object": "model",
                "owned_by": "local"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_stub():

    async def stream():

        text = "Hello from the EMTAC stub AI."

        chunk1 = {
            "id": "chatcmpl-stub",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "emtac-local",
            "choices": [
                {
                    "delta": {"content": text},
                    "index": 0,
                    "finish_reason": None
                }
            ]
        }

        yield "data: " + json.dumps(chunk1) + "\n\n"

        await asyncio.sleep(0.1)

        chunk2 = {
            "id": "chatcmpl-stub",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "emtac-local",
            "choices": [
                {
                    "delta": {},
                    "index": 0,
                    "finish_reason": "stop"
                }
            ]
        }

        yield "data: " + json.dumps(chunk2) + "\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
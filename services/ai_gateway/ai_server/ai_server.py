from fastapi import FastAPI, Request
import logging

app = FastAPI()

logging.basicConfig(level=logging.INFO)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):

    payload = await request.json()

    messages = payload.get("messages", [])

    logging.info("Received chat request")

    reply_text = "Local AI response placeholder"

    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": reply_text
                },
                "finish_reason": "stop"
            }
        ]
    }
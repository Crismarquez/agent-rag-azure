from datetime import datetime
import uuid
import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from agents.rag.base import RAGAgent
from schemas.conversation import InputChat, ResponseRAG

router = APIRouter(prefix="/api/v1")

@router.post("/chat", tags=["assistant"])
async def process_data(
    request: Request, input: InputChat
) -> dict:
    call_id = str(uuid.uuid4())
    user_id = input.user_id
    conversation_id = input.conversation_id
    metadata = {
        "user_id": user_id,
        "conversation_id": conversation_id
        }

    assistant = RAGAgent(llm_provider="azure", retrieval_args={})
    predict = await assistant.run(history=input.history, metadata=metadata)

    response = {
        "id": call_id,
        "user_id": user_id,
        "conversation_id": conversation_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "response": predict["messages"][-1].content,
    }

    return JSONResponse(status_code=200, content=response)

@router.post("/streamchat", tags=["rag"])
async def process_data(
    request: Request, input: InputChat
) -> dict:
    call_id = str(uuid.uuid4())
    user_id = input.user_id
    conversation_id = input.conversation_id
    metadata = {
        "user_id": user_id,
        "conversation_id": conversation_id
        }

    agent = RAGAgent(llm_provider="azure", retrieval_args={})

    async def event_generator():
        async for event in agent.stream_run(input.history, metadata):
            type_event = event['type']
            if type_event == "custom":
                yield f"event: {event['type']}\n"
                yield f"data: {json.dumps(event['data'])}\n\n"
            elif type_event == "messages":
                yield f"event: {event['type']}\n"
                yield f"data: {json.dumps(event['data'][0].content)}\n\n"

    return StreamingResponse(
        event_generator(), 
        media_type="text/event-stream", 
        headers={"Cache-Control": "no-cache","Connection": "keep-alive"}
        )

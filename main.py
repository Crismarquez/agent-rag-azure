import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from routers.ragagent import router as ragrouter

app = FastAPI()
app.title = "Backend RAG Agents"
app.version = "0.0.1"

app.include_router(ragrouter)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Cache-Control", "Content-Type", "Connection"]
)

@app.get("/", tags=["home"])
def message():
    return HTMLResponse("<h1>Service: RAG Agents</h1>")

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000)
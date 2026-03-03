from fastapi import FastAPI
from api.schemas import QueryRequest, QueryResponse
from orchestrator.classifier import classifier_chain
from orchestrator.router import route_to_agent

app = FastAPI(title="Multi-Agent AI Service")

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):

    classification = await classifier_chain.ainvoke(
        {"question": request.question}
    )

    agent_result = await route_to_agent(
        classification["domain"],
        request.question
    )

    return QueryResponse(
        classification=classification,
        result=agent_result
    )
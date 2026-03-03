from fastapi import APIRouter

from src.api.schema import EvaluateResponse
from src.eval.runner import run_evaluation

router = APIRouter(prefix="/api")

@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate():
    response = await run_evaluation()
    return EvaluateResponse(
        results=response[0],
        metrics=response[1]
    )
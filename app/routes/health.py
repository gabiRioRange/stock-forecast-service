from fastapi import APIRouter

from app.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Verifica se a API está no ar."""
    return HealthResponse(status="ok")

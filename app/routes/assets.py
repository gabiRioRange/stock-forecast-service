from fastapi import APIRouter

from app.config import SUPPORTED_TICKERS
from app.schemas import AssetListResponse

router = APIRouter()


@router.get("/assets", response_model=AssetListResponse)
def list_assets():
    """Lista tickers suportados pela API."""
    return AssetListResponse(tickers=SUPPORTED_TICKERS)

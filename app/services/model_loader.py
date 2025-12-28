from functools import lru_cache

from ml.models import load_model


@lru_cache(maxsize=32)
def get_model(ticker: str, model_type: str = "random_forest"):
    """Carrega modelo em cache para evitar reload em cada request."""
    return load_model(ticker, model_type)


def clear_model_cache():
    """Limpa cache de modelos (útil para testes)."""
    get_model.cache_clear()

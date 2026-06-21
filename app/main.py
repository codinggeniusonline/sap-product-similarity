"""
Part 2: FastAPI microservice.

GET /find_similar_products?product_id=...&num_similar=...

Design notes
------------
- The similarity engine builds a feature matrix over the whole dataset, which
  is expensive to do per-request. We build it ONCE at startup (lifespan) and
  reuse it, so requests are fast.
- We translate domain errors into proper HTTP codes:
    404  product_id not in dataset
    422  invalid num_similar (FastAPI validates automatically, but we also
         guard explicitly)
    500  anything unexpected
- A /health endpoint is included so Kubernetes liveness/readiness probes work.
"""

from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Query

from .similarity import get_engine, find_similar_products


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the engine at startup so the first request isn't slow.
    get_engine()
    yield


app = FastAPI(title="Product Similarity Service", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/find_similar_products", response_model=List[str])
def get_similar_products(
    product_id: str = Query(..., description="uniq_id of the product"),
    num_similar: int = Query(5, ge=1, le=100, description="how many to return"),
) -> List[str]:
    try:
        return find_similar_products(product_id, num_similar)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"product_id '{product_id}' not found")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"internal error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

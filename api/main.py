from fastapi import FastAPI
from api.routes import router as recognize_router

app = FastAPI(title="Iranian ALPR API")

app.include_router(recognize_router)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "iranian-alpr-api"
    }

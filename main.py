from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import qnaRouter_api  

app = FastAPI(
    title="Healthcare Chatbot api",
    description="API for solving User query",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include prediction router
app.include_router(qnaRouter_api.router, prefix="/api", tags=["QnA"])

# Root path
@app.get("/")
def read_root():
    return {"message": "Welcome to the HealthCare Chatbot API!"}

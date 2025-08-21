# Vercel-friendly entrypoint that exposes the FastAPI app at /api
# It imports the FastAPI app from main.py in the same directory
from main import app  # FastAPI instance defined in main.py

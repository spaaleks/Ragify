from fastapi import APIRouter
stub = APIRouter()
@stub.get("/api/tags")
def tags(): return []
@stub.get("/api/ps")
def ps(): return []

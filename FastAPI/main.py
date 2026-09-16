from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/hello")
async def hello():
    return {"message": "你好，这是我第一个 API"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def hello():
    return {"msg": "Hello from the separate FastAPI service in a new Conda env!"}

from fastapi import FastAPI

hello = FastAPI()


@hello.get("/")
async def root():
    return {"message": "hello"}


@hello.post("/pp")
async def post():
    return {"message": "testpost"}

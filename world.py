from fastapi import FastAPI
from enum import Enum

hello = FastAPI()


@hello.get("/")
async def root():
    return {"message": "hello"}


@hello.post("/pp")
async def post():
    return {"message": "testpost"}


@hello.get("/items/a")
async def item():
    return {"message": "22222222222222222222"}


@hello.get("/items/{item_id}")
async def item(item_id: int):
    return {"message": item_id}


class FoodEnum(str, Enum):
    apple = "apple"
    pear = "pear"
    peach = "peach"


@hello.get("/foods/{food}")
async def item(food: FoodEnum):
    if food.value == "apple":
        return {"food": food, "message": "You like apple"}

    else:
        return {"food": food, "message": "whatever"}


cars_db = [{"car_name": "cadillac"}, {"car_name": "bmw"}, {"car_name": "lada"}, {"car_name": "mercedes", }]


@hello.get("/cars")
async def cars(skip: int = 0, limit: int = 23):
    return cars_db[skip: skip + limit]

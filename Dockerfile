FROM python:3.9

WORKDIR /app
COPY requirements.txt .
RUN apt-get update && \
    apt-get install -y mesa-utils && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install git+https://github.com/Sadig05/keras-vggface.git@master
COPY . .

CMD ["uvicorn", "server:mserver", "--host", "0.0.0.0", "--port", "8000"]
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY /requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY .env /app/.env

EXPOSE 7860

CMD ["gradio", "main.py"]
FROM python:3.9-slim

WORKDIR /app

# Install system deps + upgrade pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "optiscan_api:app", "--host", "0.0.0.0", "--port", "8000"]

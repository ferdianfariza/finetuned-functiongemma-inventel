FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY model_service.py .

# HuggingFace Spaces pakai port 7860
EXPOSE 7860

CMD ["uvicorn", "model_service:app", "--host", "0.0.0.0", "--port", "7860"]
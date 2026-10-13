FROM python:3.11-slim
ENV PYTHONBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt pytest httpx
COPY src/ /app/src/
COPY tests/ /app/tests/
COPY app.py /app/
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ libffi-dev libssl-dev && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py policy.json ./
RUN mkdir -p datasets reports logs
ENV FLASK_ENV=production PYTHONPATH=/app HOST=0.0.0.0 PORT=5000
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD python -c "import requests; requests.get('http://localhost:5000/api/datasets')" || exit 1
CMD ["python", "app.py"]
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock
COPY src ./src
COPY config.yaml requirements.txt main.py daily_learn.py paper_runner.py cloud_runtime.py dashboard.py ./
CMD ["python", "-u", "cloud_runtime.py"]

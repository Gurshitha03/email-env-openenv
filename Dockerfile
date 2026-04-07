FROM python:3.10

WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1

# Run OpenEnv server (NOT inference)
CMD ["python", "server/app.py"]

FROM python:3.10

WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Environment variables (optional defaults)
ENV PYTHONUNBUFFERED=1

# Run the inference script
CMD ["python", "inference.py"]
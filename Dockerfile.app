# Use a lightweight Python base image [cite: 99]
FROM python:3.10-slim

# Set the working directory inside the container [cite: 100]
WORKDIR /app

# Install system dependencies (needed for some ML libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies [cite: 101, 102]
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire source code into the container [cite: 103]
COPY . .

# Expose ports for FastAPI (8000) and Streamlit (8501) [cite: 104]
EXPOSE 8000
EXPOSE 8501

# The actual command is handled by docker-compose.yml
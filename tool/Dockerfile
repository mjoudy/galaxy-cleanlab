# 1. Start from a Python 3.11 base system
FROM python:3.11-slim

# 2. Install required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && pip install --no-cache-dir "cleanlab[datalab]" scikit-learn xgboost \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Copy your Python script into the container
COPY cleanlab_issue_handler.py /app/
WORKDIR /app

# 4. What to run by default inside the container
ENTRYPOINT ["python", "cleanlab_issue_handler.py"]

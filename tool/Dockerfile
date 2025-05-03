FROM python:3.11-slim

WORKDIR /app

RUN pip install "cleanlab[datalab]" scikit-learn xgboost

COPY cleanlab_issue_handler.py .

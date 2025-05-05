FROM python:3.11-slim

WORKDIR /app

# Install specific versions of the required packages
RUN pip install \
    "cleanlab[datalab]==2.7.1" \
    "scikit-learn==1.6.1" \
    "xgboost==2.1.4"

COPY cleanlab_issue_handler.py .

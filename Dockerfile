# One image for Oracle Cloud and Hugging Face Spaces.
# The retention model is trained inside the image from the committed dbt snapshots,
# so the library versions used for training and serving are always identical.
# The demand forecast and pricing results are committed aggregates read as files.
FROM python:3.12-slim

ENV RETAIL_HOME=/app \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501
WORKDIR /app

COPY pyproject.toml ./
COPY src src
RUN pip install ".[app]"

COPY app app
COPY .streamlit .streamlit
COPY data/processed/ data/processed/
COPY reports/demand_metrics.json reports/demand_metrics.json
RUN python -m retail_intelligence.train

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/_stcore/health')"
CMD streamlit run app/streamlit_app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true

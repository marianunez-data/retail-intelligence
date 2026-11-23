# Run the whole laboratory in order:  make all
.PHONY: install data ingest features train demand pricing notebooks test app docker all clean

DATA_URL = https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip

install:            ## Install the package with every extra
	pip install -e ".[app,pipeline,analysis,dev]"

data:               ## Download both datasets: UCI Online Retail II (45 MB) and dunnhumby (30 MB)
	curl -L -o data/raw/online_retail_ii.zip "$(DATA_URL)"
	unzip -o data/raw/online_retail_ii.zip -d data/raw && rm data/raw/online_retail_ii.zip
	python -m retail_intelligence.ingest_pricing

ingest:             ## Step 1: Excel to typed Parquet
	python -m retail_intelligence.ingest

features:           ## Step 2: dbt build of the three feature snapshots, with data tests
	python -m retail_intelligence.features

train:              ## Step 3: validate, train and score
	python -m retail_intelligence.train

demand:             ## Notebook 02 pipeline: dbt demand models, backtest, forecast
	python -m retail_intelligence.demand

pricing:            ## Notebook 03 pipeline: dbt pricing models, elasticities, validation
	python -m retail_intelligence.pricing

notebooks:          ## Re-run the three notebooks in place (figures go to reports/figures)
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks/*.ipynb

test:               ## Unit, agent and end-to-end pipeline tests
	pytest -q

app:                ## Dashboard on http://localhost:8501
	streamlit run app/streamlit_app.py

docker:             ## Build and run the production image
	docker build -t retail-intelligence .
	docker run --rm -p 8501:8501 --env-file .env retail-intelligence

all: ingest features train demand pricing notebooks test

clean:
	rm -rf dbt/target dbt/logs data/processed/*.duckdb models/*.joblib

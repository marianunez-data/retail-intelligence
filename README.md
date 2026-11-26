# Retail Intelligence

**Three questions every retailer asks, answered end to end on real data:**

1. **Customers:** who will buy again in the next 90 days, why, and what should marketing do about it?
2. **Demand:** how many units of each product will be ordered in the next four weeks?
3. **Pricing and promotions:** how sensitive is each product to price, which promotions pay, and what should change?

Each question gets its own storytelling notebook, tested dbt models, validated models, a page in a Streamlit dashboard, and tools for an AI analyst that answers business questions with the same engines.

## Data

| Source | Content | Used for |
|---|---|---|
| [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) | 1.07 million invoice lines of a UK online wholesaler, December 2009 to December 2011 | Customer retention, demand forecasting |
| [dunnhumby Breakfast at the Frat](https://www.dunnhumby.com/source-files/) | 156 weeks of prices, promotions and unit sales for 58 products across 79 stores | Pricing and promotions |

## Stack

Python, DuckDB, dbt, scikit-learn, XGBoost, SHAP, pyfixest, Streamlit, an OpenAI-compatible LLM API, Docker, GitHub Actions.

## Roadmap

- [x] Data pipeline: ingestion and dbt models with data tests (#1)
- [x] Customer retention model: notebook, validation, SHAP (#2)
- [x] Demand forecasting: notebook, backtests, prediction intervals (#3)
- [x] Pricing and promotions: notebook, elasticities, simulator (#4)
- [ ] Streamlit app, AI analyst and deployment (#5)

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[app,pipeline,analysis,dev]" -c requirements-lock.txt
```

## License

Code: MIT. Data: see the sources above; raw data is downloaded by the pipeline and not committed.

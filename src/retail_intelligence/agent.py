"""
AI analyst: a tool-calling agent over the retention engine, the demand
forecast and the pricing model.

Works with any OpenAI-compatible chat API (Google Gemini, Groq, OpenAI, ...).
Configure with environment variables:
    LLM_API_KEY   API key (required)
    LLM_BASE_URL  e.g. https://generativelanguage.googleapis.com/v1beta/openai/
    LLM_MODEL     e.g. gemini-3.8-flash
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import pandas as pd

from retail_intelligence.engine import ACTIONS, SEGMENTS, Engine, SQLGuardError

MAX_STEPS = 6
MAX_ROWS_TO_MODEL = 25

SYSTEM_PROMPT = """You are a commercial analyst. You answer questions from marketing, purchasing
and category management with three analytical products:

1. Customer retention (Online Retail, a UK giftware wholesaler, GBP): which customers will buy
   again in the next 90 days and what to do about it.
2. Demand forecast (the same wholesaler): units each product will be ordered in the next
   four weeks, with a 10th to 90th percentile range.
3. Pricing and promotions (a different business: a US grocery chain, dunnhumby data, USD):
   price elasticities, promotion effects and a price and promotion simulator.

Rules:
- Always use the tools to get numbers. Never invent customers, products, counts or amounts.
- Retention: find_customers for lists, explain_customer for "why" about one customer.
- Demand: find_product_forecasts for products or categories, get_demand_overview for accuracy
  and totals. A forecast is a 4-week total; quote the range, not only the median.
- Pricing: get_pricing_overview for elasticities and promotion lifts by category,
  simulate_price_scenario for "what if" questions. An elasticity of -2 means a 1% price increase
  loses about 2% of units. Profit depends on an assumed gross margin (30% unless the user gives one);
  always state the margin used.
- run_sql for aggregations the other tools cannot express.
- Never mix the businesses: retention and demand are the wholesaler, pricing is the grocery chain.
- Explanations of customer scores come from SHAP values in log-odds; describe them in plain language.
- Be concise and business-focused: lead with the answer, then 2 to 4 supporting points,
  then a recommended next step when relevant.
- If a question is outside these topics, say so briefly.
"""

RETENTION_TOOLS = [
    {"type": "function", "function": {
        "name": "get_summary",
        "description": "Customer base KPIs: total customers, expected repurchasers, customers per "
                       "segment, customers and past spend per recommended action.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "find_customers",
        "description": "List customers matching filters, sorted. Use for target lists.",
        "parameters": {"type": "object", "properties": {
            "segment": {"type": "string", "enum": SEGMENTS},
            "action": {"type": "string", "enum": ACTIONS},
            "value_tier": {"type": "string", "enum": ["High value", "Standard"]},
            "uk_only": {"type": "boolean", "description": "true = UK only, false = outside UK"},
            "min_spend": {"type": "number", "description": "Minimum total spent in GBP"},
            "min_recency_days": {"type": "integer", "description": "Minimum days since last order"},
            "sort_by": {"type": "string", "enum": ["probability", "monetary", "recency_days", "frequency"]},
            "ascending": {"type": "boolean"},
            "limit": {"type": "integer", "description": "Max rows, default 20, max 500"}}}}},
    {"type": "function", "function": {
        "name": "explain_customer",
        "description": "Explain one customer's score with SHAP: probability, segment, action and "
                       "the features that raise or lower it, compared with a typical customer.",
        "parameters": {"type": "object", "properties": {
            "customer_id": {"type": "integer"}}, "required": ["customer_id"]}}},
    {"type": "function", "function": {
        "name": "run_sql",
        "description": "Run one read-only DuckDB SELECT query. Results are limited to 200 rows.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_model_performance",
        "description": "Model quality on a holdout set and on a later period (out-of-time): AUC, "
                       "Brier score, share of repurchasers reached by contacting the top 20%, lift, "
                       "and the most important features.",
        "parameters": {"type": "object", "properties": {}}}},
]


DEMAND_TOOLS = [
    {"type": "function", "function": {
        "name": "get_demand_overview",
        "description": "Demand forecast summary for the wholesaler: forecast window, total units by category "
                       "(median forecast vs the last 4 weeks), backtest accuracy against simple baselines "
                       "and interval coverage.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "find_product_forecasts",
        "description": "Four-week demand forecasts (p10, p50, p90 units) for wholesaler products, filtered "
                       "by text in the description or product id, and/or by category.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "Words in the description, or a product id"},
            "category": {"type": "string"},
            "sort_by": {"type": "string", "enum": ["p50", "p90", "last_4_weeks"]},
            "limit": {"type": "integer", "description": "Max rows, default 15"}}}}},
]

PRICING_TOOLS = [
    {"type": "function", "function": {
        "name": "get_pricing_overview",
        "description": "Grocery chain pricing results: average base price and discount elasticities by "
                       "category, promotion lifts (circular feature, display, shelf tag), the discount "
                       "threshold, model validation, and the most and least price-sensitive products.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "simulate_price_scenario",
        "description": "Simulate one grocery product's weekly chain units, revenue and gross profit for a "
                       "base price change and/or a promotion, compared with a regular week.",
        "parameters": {"type": "object", "properties": {
            "product": {"type": "string", "description": "UPC, or words from the product description"},
            "base_price_change": {"type": "number", "description": "e.g. 0.05 for +5%, limited to -0.10..0.10"},
            "discount": {"type": "number", "description": "Temporary discount, e.g. 0.2 for 20% off"},
            "feature": {"type": "boolean", "description": "Featured in the weekly circular"},
            "display": {"type": "boolean", "description": "On an in-store display"},
            "gross_margin": {"type": "number", "description": "Assumed gross margin at today's price, default 0.30"}},
            "required": ["product"]}}},
]


def tools_for(engine: Engine) -> list[dict]:
    """Tool definitions available for this deployment; the SQL tool lists every registered table."""
    tools = [dict(t) for t in RETENTION_TOOLS]
    for t in tools:
        if t["function"]["name"] == "run_sql":
            fn = dict(t["function"])
            fn["description"] = fn["description"] + "\n" + engine.sql_schema
            t["function"] = fn
    if engine.demand is not None:
        tools += DEMAND_TOOLS
    if engine.pricing is not None:
        tools += PRICING_TOOLS
    return [_portable(t) for t in tools]


def _portable(tool: dict) -> dict:
    """Drop empty parameter objects: Gemini's OpenAI-compatible API rejects an object schema
    with no properties, while OpenAI accepts a function declared without parameters."""
    fn = tool["function"]
    if not fn.get("parameters", {}).get("properties"):
        fn = {k: v for k, v in fn.items() if k != "parameters"}
        return {"type": "function", "function": fn}
    return tool


TOOLS = RETENTION_TOOLS   # backwards-compatible name


@dataclass
class Step:
    tool: str
    args: dict
    ok: bool
    table: pd.DataFrame | None = None
    note: str = ""


@dataclass
class AgentResult:
    answer: str
    steps: list[Step] = field(default_factory=list)

    @property
    def tables(self) -> list[pd.DataFrame]:
        return [s.table for s in self.steps if s.table is not None and not s.table.empty]


def _frame_payload(df: pd.DataFrame) -> dict:
    head = df.head(MAX_ROWS_TO_MODEL)
    return {"row_count": int(len(df)), "rows_shown": int(len(head)),
            "rows": json.loads(head.to_json(orient="records", double_precision=4))}


LIST_COLS = ["CustomerID", "probability", "segment", "value_tier", "action",
             "recency_days", "frequency", "monetary"]


def execute_tool(engine: Engine, name: str, args: dict) -> tuple[dict, Step]:
    """Run one tool. Returns (payload for the model, step for the UI)."""
    try:
        if name == "get_summary":
            return engine.summary(), Step(name, args, True)

        if name == "find_customers":
            args = {k: v for k, v in args.items() if v is not None}
            args.setdefault("limit", 20)
            df = engine.find(**args)[LIST_COLS]
            return _frame_payload(df), Step(name, args, True, table=df)

        if name == "explain_customer":
            row = engine.customer(args["customer_id"])
            if row is None:
                return {"error": f"Customer {args['customer_id']} not found."}, \
                    Step(name, args, False, note="Customer not found")
            exp, base = engine.explain(row)
            payload = {
                "customer_id": int(row["CustomerID"]),
                "probability": round(float(row["probability"]), 4),
                "average_customer_probability": round(base, 4),
                "segment": row["segment"], "value_tier": row["value_tier"], "action": row["action"],
                "drivers": [{"feature": r.label,
                             "value": None if pd.isna(r.value) else round(r.value, 2),
                             "typical_customer": round(r.typical, 2),
                             "shap_log_odds": round(r.shap, 3)}
                            for r in exp.itertuples()],
            }
            table = exp[["label", "value", "typical", "shap"]].rename(columns={
                "label": "Feature", "value": "Customer", "typical": "Typical", "shap": "SHAP (log-odds)"})
            return payload, Step(name, args, True, table=table)

        if name == "run_sql":
            df = engine.sql(args["query"])
            return _frame_payload(df), Step(name, args, True, table=df)

        if name == "get_model_performance":
            m = engine.metrics
            payload = {"model": m["model"], "holdout": m["holdout"], "out_of_time": m["out_of_time"],
                       "logistic_regression_holdout_auc": m["holdout_logistic"]["auc"],
                       "recency_rule_holdout_auc": m["holdout_recency_rule"]["auc"],
                       "training_repurchase_rate": m["training_snapshot"]["repurchase_rate"],
                       "top_features": engine.importance[["label", "mean_abs_shap"]].head(5)
                                             .round(3).to_dict("records")}
            return payload, Step(name, args, True)

        if name == "get_demand_overview":
            d = engine.demand
            m = d.metrics
            cats = d.by_category().round(1).reset_index()
            payload = {"forecast_window": m["forecast_window"], "products": m["products"],
                       "total_p50": round(float(d.forecast.p50.sum())),
                       "total_last_4_weeks": round(float(d.forecast.last_4_weeks.sum())),
                       "backtest": {"period": m["backtest_period"], "methods": m["methods"],
                                    "best_baseline": m["best_baseline"],
                                    "interval_coverage_calibrated": m["interval_coverage_calibrated"]},
                       "by_category": cats.to_dict("records")}
            return payload, Step(name, args, True, table=cats)

        if name == "find_product_forecasts":
            args = {k: v for k, v in args.items() if v not in (None, "")}
            df = engine.demand.find(args.get("text"), args.get("category"), args.get("sort_by", "p50"),
                                    args.get("limit", 15))
            df = df[["product_id", "description", "category", "last_4_weeks", "p10", "p50", "p90"]]
            return _frame_payload(df), Step(name, args, True, table=df)

        if name == "get_pricing_overview":
            pm = engine.pricing
            prod = pm.products.sort_values("elasticity_base_pooled")
            cols = ["description", "category", "base_price", "elasticity_base_pooled", "recommendation"]
            payload = {"category_elasticities": pm.model["category_elasticities"],
                       "promotion_lift_by_category": {c: {t: v["lift"] for t, v in terms.items()}
                                                      for c, terms in pm.model["promo_effects"].items()},
                       "discount_threshold": "Discounts under 10% showed no measurable lift",
                       "validation_holdout_wape": {k: round(v["wape"], 3) for k, v in
                                                   pm.model["validation"]["all_weeks"].items()},
                       "most_price_sensitive": prod.head(5)[cols].round(2).to_dict("records"),
                       "least_price_sensitive": prod.tail(5)[cols].round(2).to_dict("records"),
                       "default_gross_margin": pm.model["settings"]["default_gross_margin"]}
            table = (pd.DataFrame(pm.model["category_elasticities"]).T.reset_index(names="Category")
                       .rename(columns={"elasticity_base": "Base price elasticity",
                                        "elasticity_promo": "Discount elasticity",
                                        "between_product_sd_base": "Spread between products"}).round(2))
            return payload, Step(name, args, True, table=table)

        if name == "simulate_price_scenario":
            pm = engine.pricing
            key = str(args.get("product", ""))
            row = pm.product(key)
            if row is None:
                hits = pm.find(key)
                if hits.empty:
                    return {"error": f"No grocery product matches '{key}'."}, Step(name, args, False, note="No match")
                if len(hits) > 1 and not hits.description.str.fullmatch(key, case=False).any():
                    return ({"error": "Several products match; ask the user which one or pass a UPC.",
                             "matches": hits[["upc", "description", "category"]].head(10).to_dict("records")},
                            Step(name, args, False, note="Ambiguous product"))
                row = hits.iloc[0]
            kwargs = {k: args[k] for k in ["base_price_change", "discount", "feature", "display", "gross_margin"]
                      if args.get(k) is not None}
            r = pm.simulate(row.upc, **kwargs)
            payload = {"product": row.description, "upc": row.upc, "category": row.category,
                       "assumed_gross_margin": kwargs.get("gross_margin", pm.model["settings"]["default_gross_margin"]),
                       "elasticity_base": round(float(row.elasticity_base_pooled), 2),
                       **{k: round(float(v), 4) for k, v in r.items()}}
            table = pd.DataFrame([
                {"Measure": "Units per week", "Regular week": r["units_baseline"], "Scenario": r["units"],
                 "Change": r["units_change"]},
                {"Measure": "Revenue (USD)", "Regular week": r["revenue_baseline"], "Scenario": r["revenue"],
                 "Change": r["revenue_change"]},
                {"Measure": "Gross profit (USD)", "Regular week": r["gross_profit_baseline"],
                 "Scenario": r["gross_profit"], "Change": r["gross_profit_change"]},
            ]).round(3)
            return payload, Step(name, args, True, table=table)

        return {"error": f"Unknown tool {name}"}, Step(name, args, False, note="Unknown tool")

    except SQLGuardError as e:
        return {"error": f"Query rejected: {e}"}, Step(name, args, False, note=f"Rejected: {e}")
    except Exception as e:  # report tool errors back to the model so it can recover
        return {"error": str(e)}, Step(name, args, False, note=str(e))


def make_client():
    """OpenAI-compatible client from environment variables, or None if not configured."""
    key = os.getenv("LLM_API_KEY")
    if not key:
        return None
    from openai import OpenAI
    return OpenAI(api_key=key, base_url=os.getenv("LLM_BASE_URL") or None)


def model_name() -> str:
    return os.getenv("LLM_MODEL", "gemini-3.8-flash")


def _echo_call(call) -> dict:
    """A tool call to send back in the next request. Provider-specific fields are kept unchanged:
    Gemini 3 attaches extra_content.google.thought_signature and rejects the follow-up without it."""
    out = {"id": call.id, "type": "function",
           "function": {"name": call.function.name, "arguments": call.function.arguments}}
    out.update(getattr(call, "model_extra", None) or {})
    return out


def run_agent(question: str, engine: Engine, client, model: str,
              history: list[dict] | None = None) -> AgentResult:
    tools = tools_for(engine)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in (history or [])[-6:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": question})
    steps: list[Step] = []

    for _ in range(MAX_STEPS):
        resp = client.chat.completions.create(model=model, messages=messages,
                                              tools=tools, temperature=0)
        msg = resp.choices[0].message
        calls = msg.tool_calls or []
        if not calls:
            return AgentResult(msg.content or "", steps)

        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [_echo_call(c) for c in calls]})
        for c in calls:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            payload, step = execute_tool(engine, c.function.name, args)
            steps.append(step)
            messages.append({"role": "tool", "tool_call_id": c.id,
                             "content": json.dumps(payload, default=str)})

    return AgentResult("I could not finish this analysis within the step limit. "
                       "Try a more specific question.", steps)

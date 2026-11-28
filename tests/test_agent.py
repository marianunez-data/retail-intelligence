"""Agent loop tests with a scripted fake LLM: no API key or network needed."""
import json
from types import SimpleNamespace as NS

from retail_intelligence.agent import execute_tool, run_agent


class FakeLLM:
    """Plays back a fixed list of responses: tool calls first, then a final answer."""

    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return NS(choices=[NS(message=self.script.pop(0))])


def tool_call(name, args, i=0):
    return NS(id=f"c{i}", function=NS(name=name, arguments=json.dumps(args)))


def test_agent_calls_tools_then_answers(engine):
    llm = FakeLLM([
        NS(content="", tool_calls=[tool_call("find_customers", {"segment": "At risk", "limit": 5})]),
        NS(content="Here are 5 at-risk customers.", tool_calls=None),
    ])
    res = run_agent("Who is at risk?", engine, llm, "fake-model")
    assert res.answer == "Here are 5 at-risk customers."
    assert [s.tool for s in res.steps] == ["find_customers"]
    assert len(res.tables[0]) == 5
    tool_msg = llm.calls[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and json.loads(tool_msg["content"])["row_count"] == 5


def test_rejected_sql_is_reported_to_model(engine):
    llm = FakeLLM([
        NS(content="", tool_calls=[tool_call("run_sql", {"query": "DROP TABLE customers"})]),
        NS(content="I can only run read-only queries.", tool_calls=None),
    ])
    res = run_agent("Delete everything", engine, llm, "fake-model")
    assert not res.steps[0].ok
    assert "rejected" in llm.calls[1]["messages"][-1]["content"].lower()


def test_step_limit_stops_runaway_loop(engine):
    loop = [NS(content="", tool_calls=[tool_call("get_summary", {}, i)]) for i in range(10)]
    res = run_agent("Loop", engine, FakeLLM(loop), "fake-model")
    assert "step limit" in res.answer


def test_explain_unknown_customer(engine):
    payload, step = execute_tool(engine, "explain_customer", {"customer_id": -1})
    assert "error" in payload and not step.ok


def test_explain_known_customer(engine):
    cid = int(engine.scored["CustomerID"].iloc[0])
    payload, step = execute_tool(engine, "explain_customer", {"customer_id": cid})
    assert payload["customer_id"] == cid and len(payload["drivers"]) == len(engine.features)


def test_all_tool_groups_offered_when_outputs_exist(engine):
    from retail_intelligence.agent import tools_for
    names = {t["function"]["name"] for t in tools_for(engine)}
    assert {"find_customers", "find_product_forecasts", "simulate_price_scenario"} <= names
    sql = next(t for t in tools_for(engine) if t["function"]["name"] == "run_sql")
    assert "product_forecast" in sql["function"]["description"]


def test_demand_tools(engine):
    payload, step = execute_tool(engine, "find_product_forecasts", {"text": "heart"})
    assert step.ok and payload["row_count"] == 1 and payload["rows"][0]["p50"] == 20.0
    payload, step = execute_tool(engine, "get_demand_overview", {})
    assert step.ok and payload["total_p50"] == 28


def test_price_simulation_tool(engine):
    payload, step = execute_tool(engine, "simulate_price_scenario",
                                 {"product": "pizza", "discount": 0.2, "display": True})
    assert step.ok and payload["product"] == "DIGRN PEPP PIZZA"
    assert payload["units_change"] > 0 and payload["assumed_gross_margin"] == 0.3


def test_price_simulation_unknown_product(engine):
    payload, step = execute_tool(engine, "simulate_price_scenario", {"product": "caviar"})
    assert not step.ok and "error" in payload


def test_sql_can_join_new_tables(engine):
    df = engine.sql("SELECT category, avg(elasticity_base_pooled) AS e FROM product_pricing GROUP BY 1")
    assert len(df) == 2

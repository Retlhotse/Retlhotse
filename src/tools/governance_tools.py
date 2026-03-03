from langchain.tools import tool
from db.postgres import run_query

@tool
async def get_governance_rules() -> str:
    return await run_query("SELECT * FROM governance_rules", {})

@tool
async def get_compliance_records(app_id: int) -> str:
    sql = "SELECT * FROM compliance WHERE app_id = :id"
    return await run_query(sql, {"id": app_id})

@tool
async def get_rule_apps(rule_id: int) -> str:
    sql = "SELECT * FROM rule_app_map WHERE rule_id = :id"
    return await run_query(sql, {"id": rule_id})
from langchain.tools import tool
from db.postgres import run_query

@tool
async def query_dq_issues(application: str) -> str:
    """Fetch data quality issues for an application."""
    sql = "SELECT * FROM dq_issues WHERE app_name = :app"
    return await run_query(sql, {"app": application})

@tool
async def query_impacted_apps(issue_id: int) -> str:
    sql = "SELECT * FROM impacted_apps WHERE issue_id = :id"
    return await run_query(sql, {"id": issue_id})

@tool
async def query_improvement_actions(application: str) -> str:
    sql = "SELECT * FROM improvement_actions WHERE app_name = :app"
    return await run_query(sql, {"app": application})
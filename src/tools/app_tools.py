from langchain.tools import tool
from db.postgres import run_query

@tool
async def search_apps(keyword: str) -> str:
    sql = "SELECT * FROM applications WHERE name ILIKE :kw"
    return await run_query(sql, {"kw": f"%{keyword}%"})

@tool
async def get_app_profile(app_id: int) -> str:
    sql = "SELECT * FROM applications WHERE id = :id"
    return await run_query(sql, {"id": app_id})

@tool
async def get_app_dependencies(app_id: int) -> str:
    sql = "SELECT * FROM dependencies WHERE app_id = :id"
    return await run_query(sql, {"id": app_id})
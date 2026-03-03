from agents.base import build_agent
from tools.dq_tools import query_dq_issues, query_impacted_apps

def build_dq_investigation_agent():
    system_prompt = """
You investigate data quality issues.
Identify root causes and impacted applications.
"""
    return build_agent(system_prompt, [
        query_dq_issues,
        query_impacted_apps
    ])
from agents.base import build_agent
from tools.dq_tools import query_improvement_actions

def build_dq_improvement_agent():
    system_prompt = """
You recommend actions to improve data quality.
"""
    return build_agent(system_prompt, [
        query_improvement_actions
    ])
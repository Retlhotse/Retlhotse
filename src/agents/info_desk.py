from agents.base import build_agent
from tools.search_tool import search_user_guide

def build_info_desk_agent():
    system_prompt = """
You are the Info Desk agent.
Use the search tool to answer website navigation questions.
Always search before answering.
"""
    return build_agent(system_prompt, [search_user_guide])
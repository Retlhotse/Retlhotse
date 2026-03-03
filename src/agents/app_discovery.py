from agents.base import build_agent
from tools.app_tools import (
    get_app_profile,
    search_apps,
    get_app_dependencies
)

def build_app_discovery_agent():
    system_prompt = """
You provide a consolidated application profile including
purpose and dependencies.
"""
    return build_agent(system_prompt, [
        search_apps,
        get_app_profile,
        get_app_dependencies
    ])
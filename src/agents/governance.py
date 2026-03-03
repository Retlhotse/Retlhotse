from agents.base import build_agent
from tools.governance_tools import (
    get_governance_rules,
    get_compliance_records,
    get_rule_apps
)

def build_governance_agent():
    system_prompt = """
You answer governance and compliance questions.
Assess risk and rule violations.
"""
    return build_agent(system_prompt, [
        get_governance_rules,
        get_compliance_records,
        get_rule_apps
    ])
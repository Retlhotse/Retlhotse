from agents.info_desk import build_info_desk_agent
from agents.dq_investigation import build_dq_investigation_agent
from agents.dq_improvement import build_dq_improvement_agent
from agents.app_discovery import build_app_discovery_agent
from agents.governance import build_governance_agent

async def route_to_agent(domain: str, question: str):

    agent_map = {
        "info_desk": build_info_desk_agent(),
        "dq_investigation": build_dq_investigation_agent(),
        "dq_improvement": build_dq_improvement_agent(),
        "app_discovery": build_app_discovery_agent(),
        "governance": build_governance_agent(),
    }

    agent = agent_map[domain]

    result = await agent.ainvoke({"input": question})

    return {
        "domain": domain,
        "answer": result["output"],
        "metadata": {}
    }
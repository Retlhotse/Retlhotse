from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from config import settings

llm = ChatOpenAI(
    api_key=settings.OPENAI_API_KEY,
    model=settings.OPENAI_MODEL,
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a classifier.

Classify the user question into one of:
- info_desk
- dq_investigation
- dq_improvement
- app_discovery
- governance

Return JSON:
{
  "domain": "...",
  "confidence": 0-1
}
"""),
    ("human", "{question}")
])

classifier_chain = prompt | llm | JsonOutputParser()
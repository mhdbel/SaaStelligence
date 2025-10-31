# web/app.py

from fastapi import FastAPI
from SaaStelligence.agents.saastelligence_agent import SAAStelligenceAgent

app = FastAPI()
agent = SAAStelligenceAgent()

@app.post("/generate-ad")
async def generate_ad(query: str, user_id: str = None, last_action: str = None):
    return agent.run(query, user_id, last_action)

@app.get("/report")
async def get_report():
    return agent.report_performance()
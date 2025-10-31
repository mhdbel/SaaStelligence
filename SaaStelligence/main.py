# main.py

from SaaStelligence.agents.saastelligence_agent import SAAStelligenceAgent

if __name__ == "__main__":
    agent = SAAStelligenceAgent()
    
    # Simulated user input
    user_query = "How can I automate my sales team’s tasks?"
    result = agent.run(user_query, user_id='12345', last_action='email_submitted')
    
    print("🚀 SAAStelligence Response:")
    for k, v in result.items():
        print(f"{k}: {v}")
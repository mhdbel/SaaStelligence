# SAAStelligence - AGI-Level SaaS CPA Engine

> A fully autonomous AI agent designed to dominate the trillion-dollar SaaS lead gen market.

## Features

- **Modular Component Architecture**: Core functionalities are broken down into manageable, reusable components.
- **Dynamic Ad Generation**: Real-time ad creative generation based on user intent.
- **Configurable A/B Testing**: Foundational support for A/B testing ad creatives and funnel routes.
- **Behavioral Funnel Routing**: Directs users through different funnels based on detected intent and A/B test configurations.
- **Automated Bid Adjustment**: Basic CVR-based bid adjustment logic.
- **Flexible Retargeting Orchestration**: Defines retargeting tasks via configuration, moving away from hardcoded URLs.
- **Behavioral Scoring**: Initial framework for tracking user actions and calculating a behavioral score.
- **Self-learning via Feedback Loops**: (Retained from original - implementation details not covered in recent tasks)

## Project Architecture

The SAAStelligence agent is built around a modular architecture, with core functionalities encapsulated in distinct components found within the `SaaStelligence/components/` directory. This design promotes separation of concerns and easier maintenance.

Key components include:

-   **`IntentDetector`**: Analyzes user queries to determine their underlying intent (e.g., 'workflow_automation', 'sales_team_efficiency').
-   **`AdGenerator`**: Dynamically generates ad copy based on the detected intent. Supports A/B testing of different ad prompt templates.
-   **`FunnelRouter`**: Routes users to appropriate funnels based on their intent. Supports A/B testing of different funnel paths for the same intent.
-   **`BidAdjuster`**: Adjusts advertising bids based on factors like predicted conversion rates (CVR).
-   **`RetargetingManager`**: Orchestrates retargeting actions based on user behavior (e.g., 'email_submitted', 'form_abandoned'). Actions are defined in the configuration and can target multiple platforms.
-   **`BehavioralScorer`**: Tracks user actions and maintains a behavioral score for each user, providing insights into user engagement.

Many of these components are configuration-driven, with their parameters and behaviors defined in `SaaStelligence/config/config.py`. This allows for easier tuning and experimentation without modifying core component logic.

The main agent logic resides in `SaaStelligence/agents/saastelligence_agent.py`, which coordinates these components to process user interactions and make decisions.

## Getting Started

1.  Clone repo: `git clone https://github.com/your-org/saastelligence.git`
2.  Install dependencies: `pip install -r SaaStelligence/requirements.txt` (Note: Ensure your environment has enough disk space, as full dependencies can be large).
3.  Configure API keys and other settings in a `.env` file at the root of the `SaaStelligence` project directory (refer to `SaaStelligence/config/config.py` for variables loaded from `.env`).
4.  Run the main application: `python SaaStelligence/main.py` or launch the web API via `uvicorn SaaStelligence.web.app:app --reload --root-path /app/SaaStelligence` (adjust path if needed).

## Technologies Used

-   Python
-   FastAPI (for web API)
-   TensorFlow (for intent detection model)
-   LangChain (for LLM interactions)
-   Pandas, NumPy, Scikit-learn (for data handling and ML)
-   Pytest (for unit testing)
-   Configuration is managed via Python scripts (`config.py`) and environment variables (`.env`).

*(Self-learning via feedback loops and specific integrations like Google Looker Studio are high-level goals and their detailed implementation might vary based on ongoing development.)*

# SAAStelligence - AGI-Level SaaS CPA Engine

> A fully autonomous AI agent designed to dominate the trillion-dollar SaaS lead gen market.

## Features

- Real-time ad generation
- Behavioral funnel routing
- Automated bidding
- Retargeting orchestration
- Self-learning via feedback loops

## Getting Started

1. Clone repo: `git clone https://github.com/your-org/saastelligence.git` 
2. (Optional) Install integrations: `pip install -r requirements.txt`
3. Configure API keys and optional integrations in `.env` file
4. Review or extend the canonical dataset at `SaaStelligence/data/conversions.csv` (required columns: `query_text`, `intent`, `converted`, optional performance metrics such as `clicks`, `impressions`, `cost`).
5. Train the intent model: `python SaaStelligence/models/train_intent_model.py`
6. Run automated checks: `python -m unittest discover -s tests`
7. Launch the CLI agent: `python -m SaaStelligence.main`

## Technologies Used

- Python standard library, optional LangChain/OpenAI bindings, Google Sheets helpers

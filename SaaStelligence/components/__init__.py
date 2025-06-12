# SaaStelligence/components/__init__.py
"""
This package contains the various components that make up the SAAStelligence agent.

Each component encapsulates a specific piece of functionality, such as intent detection,
ad generation, funnel routing, etc. This modular design allows for easier development,
testing, and maintenance.
"""

# Core operational components
from .intent_detector import IntentDetector
from .ad_generator import AdGenerator
from .funnel_router import FunnelRouter
from .bid_adjuster import BidAdjuster
from .retargeting_manager import RetargetingManager
from .behavioral_scorer import BehavioralScorer

# Placeholder components for future features
from .email_nurturer import EmailNurturer
from .webinar_scheduler import WebinarScheduler
from .customer_success_automator import CustomerSuccessAutomator

# Optionally, define __all__ to specify what is exported when 'from . import *' is used.
# This also helps linters and IDEs understand the public interface of the package.
__all__ = [
    "IntentDetector",
    "AdGenerator",
    "FunnelRouter",
    "BidAdjuster",
    "RetargetingManager",
    "BehavioralScorer",
    "EmailNurturer",
    "WebinarScheduler",
    "CustomerSuccessAutomator",
]

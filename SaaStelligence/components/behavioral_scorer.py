# components/behavioral_scorer.py
"""
Manages behavioral scoring for users based on their actions.

This component tracks various user actions, assigns scores to these actions
based on a configurable mapping, and maintains a cumulative score for each user.
Scores are currently stored in-memory.
"""
from SaaStelligence.config.config import CONFIG

class BehavioralScorer:
    """
    Tracks user actions and calculates behavioral scores.
    Scores are stored in-memory and are based on configured action point values.
    """
    def __init__(self):
        """
        Initializes the BehavioralScorer.

        Loads the action-to-score mapping from the application configuration
        and initializes an in-memory dictionary to store user scores.
        """
        self.action_scores_config = CONFIG.BEHAVIORAL_ACTION_SCORES
        self.user_scores = {}  # In-memory storage for user scores

    def track_action(self, user_id: str, action_type: str, details: dict = None):
        """
        Tracks a specified action for a user and updates their behavioral score.

        If the `user_id` is None, the action is not tracked. Points for the
        `action_type` are retrieved from the configuration. If the `action_type`
        is not found in the configuration, it defaults to 0 points.

        Args:
            user_id: The unique identifier for the user.
            action_type: A string representing the type of action performed
                         (e.g., 'processed_query_high_intent', 'email_submitted').
            details: Optional dictionary containing additional details about the
                     action, for logging or future use.
        """
        if user_id is None:
            # In a real system, decisions on handling anonymous users would be needed.
            # For now, we simply skip tracking if no user_id is present.
            print("BehavioralScorer: user_id is None. Action tracking skipped.")
            return

        # Get points for the action_type from config, default to 0 if not found
        points = self.action_scores_config.get(action_type, 0)

        # Retrieve current score, defaulting to 0 for new users, then add points
        current_score = self.user_scores.get(user_id, 0)
        new_score = current_score + points
        self.user_scores[user_id] = new_score

        # Logging the action and score update
        print(f"BehavioralScorer: Tracked action '{action_type}' for user '{user_id}'. "
              f"Points: {points}. Old score: {current_score}. New score: {new_score}.")
        if details:
            # Log additional details if provided
            print(f"BehavioralScorer: Action details for user '{user_id}': {details}")


    def get_score(self, user_id: str) -> int:
        """
        Retrieves the current behavioral score for a given user.

        Args:
            user_id: The unique identifier for the user.

        Returns:
            The user's current behavioral score. Returns 0 if the user_id is None
            or if the user has no score tracked yet.
        """
        if user_id is None:
            # Return a default score for anonymous users or if user_id is not provided.
            return 0
        # Return the score, defaulting to 0 if user_id not in user_scores
        return self.user_scores.get(user_id, 0)

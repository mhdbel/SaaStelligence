# components/ad_generator.py
"""
Handles the generation of advertising copy using language models.

This component selects from a list of configured prompt templates (allowing for A/B testing)
and uses a language model (e.g., OpenAI's GPT) to generate ad text based on user intent.
"""
import random
from langchain import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from SaaStelligence.config.config import CONFIG
from dotenv import load_dotenv

load_dotenv() # Ensures .env variables are loaded for CONFIG, especially OPENAI_API_KEY

class AdGenerator:
    """
    Generates ad copy based on intent using configured prompt templates and an LLM.
    Supports A/B testing of different ad prompt templates.
    """
    def __init__(self):
        """
        Initializes the AdGenerator.

        Loads ad prompt templates and LLM temperature from the application configuration.
        Initializes the language model instance.
        """
        # Store the list of prompt templates from CONFIG
        self.ad_prompt_templates_list = CONFIG.AD_PROMPT_TEMPLATES
        # Initialize the language model (e.g., OpenAI).
        # The temperature is fixed for all templates in this setup.
        self.llm = OpenAI(
            openai_api_key=CONFIG.OPENAI_API_KEY,
            temperature=CONFIG.AD_LLM_TEMPERATURE
        )

    def generate_ad(self, intent: str, variation_index: int = None) -> dict:
        """
        Generates ad copy for a given intent, with optional A/B test variation selection.

        If a valid `variation_index` is provided, the corresponding template from the
        configured list is used. Otherwise, a template is chosen randomly.

        Args:
            intent: The user intent for which to generate the ad.
            variation_index: Optional specific index of the ad template to use.

        Returns:
            A dictionary containing:
                - "ad_copy": The generated advertisement text.
                - "template_index_chosen": The index of the prompt template used.
        """
        chosen_template_str = None
        chosen_template_index = -1 # Default to -1 or an indicator of random choice

        if self.ad_prompt_templates_list: # Ensure templates are loaded
            if variation_index is not None and 0 <= variation_index < len(self.ad_prompt_templates_list):
                # Use specified template if index is valid
                chosen_template_str = self.ad_prompt_templates_list[variation_index]
                chosen_template_index = variation_index
            else:
                # Randomly select a template if no valid index is provided
                chosen_template_index = random.randrange(len(self.ad_prompt_templates_list))
                chosen_template_str = self.ad_prompt_templates_list[chosen_template_index]
        else:
            # Fallback or error handling if no templates are configured
            # This case should ideally be prevented by config validation at startup
            return {
                "ad_copy": "Error: No ad prompt templates configured.",
                "template_index_chosen": -1
            }

        # Create a PromptTemplate object from the chosen string template
        current_prompt_template_obj = PromptTemplate.from_template(chosen_template_str)

        # Create the LLMChain with the selected prompt template and the initialized LLM.
        # For simplicity and flexibility with varying prompts, the chain is created per call.
        # In a high-performance scenario with fixed prompts, chains could be pre-initialized and stored.
        ad_chain = LLMChain(llm=self.llm, prompt=current_prompt_template_obj)

        # Generate the ad copy
        ad_copy_text = ad_chain.run(intent=intent)

        return {
            "ad_copy": ad_copy_text,
            "template_index_chosen": chosen_template_index
            # Optionally, one might return the full template string for logging:
            # "template_content_chosen": chosen_template_str
        }

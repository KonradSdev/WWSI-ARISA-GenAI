import os
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

class ToxicityAnalyzer:
    def __init__(self, model_name="textdetox/xlmr-large-toxicity-classifier", language="multi"):
        """
        Initializes the ToxicityAnalyzer with a specified transformer model and language.

        Args:
            model_name (str, optional): The name of the pre-trained
                toxicity analysis model from Hugging Face Transformers.
                Defaults to "unitaryai/toxic-bert-base-uncased".
            language (str, optional): The primary language of the text to be analyzed.
                Defaults to "multi" as many toxicity models are multilingual or English-centric.
        """
        self.language = language
        try:
            self.analyzer = pipeline("text-classification", model=model_name)
        except Exception as e:
            self.analyzer = None

    def analyze_toxicity(self, text):
        """
        Analyzes the toxicity of the given text.

        Args:
            text (str): The text to analyze.

        Returns:
            dict or None: A dictionary containing the toxicity label (e.g., 'toxic', 'non-toxic')
                           and the confidence score, or None if the analyzer failed.
        """
        if self.analyzer:
            try:
                result = self.analyzer(text)[0]
                return result
            except Exception as e:
                return None
        else:
            return None

    def analyze_batch_toxicity(self, text_list):
        """
        Analyzes the toxicity of a list of texts.

        Args:
            text_list (list): A list of strings to analyze.

        Returns:
            list or None: A list of dictionaries, where each dictionary contains
                          the toxicity label and score for the corresponding text,
                          or None if the analyzer failed.
        """
        if self.analyzer:
            try:
                results = self.analyzer(text_list)
                return results
            except Exception as e:
                return None
        else:
            return None

    def is_toxic(self, text, threshold=0.8):
        """
        Checks if the text is considered toxic based on the model's prediction
        and a given confidence threshold.

        Args:
            text (str): The text to analyze.
            threshold (float, optional): The minimum confidence score for considering
                                        the text as toxic. Defaults to 0.8.

        Returns:
            bool or None: True if the text is toxic with sufficient confidence,
                           False otherwise, or None if analysis failed.
        """
        result = self.analyze_toxicity(text)
        if result and result['label'] == 'toxic' and result['score'] >= threshold:
            return True
        elif result and result['label'] == 'non-toxic' and result['score'] < (1 - threshold):
            return False # Consider non-toxic if confidence is high
        return None

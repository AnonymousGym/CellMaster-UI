from agents.hypothesis_agent.base_hypothesis_agent import BaseHypothesisAgent
import openai
from config.settings import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

class PhotosynthesisHypothesisAgent(BaseHypothesisAgent):
    def __init__(self, hypothesis):
        super().__init__(hypothesis)

    def refine_hypothesis(self, evaluation_result=None):
        content = f"Literature Summary:\n\n{self.summary}\n\nCurrent Hypothesis:\n{self.hypothesis}\n\n"
        if evaluation_result:
            content += f"Evaluation Result:\n\n{evaluation_result}\n\n"
        content += "Please refine the hypothesis based on the literature summary"
        if evaluation_result:
            content += " and evaluation result"
        content += ", focusing on photosynthesis-related insights."

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a research assistant specializing in photosynthesis. Based on the summarized literature and evaluation result (if provided), refine the given hypothesis to be more accurate and specific."},
                {"role": "user", "content": content}
            ],
            temperature=0.7
        )
        self.refined_hypothesis = response.choices[0].message['content'].strip()
        
        print("\nRefined Hypothesis:")
        print(self.refined_hypothesis)
        
    def adjust_hypothesis(self, user_feedback):
        content = f"Refined Hypothesis:\n\n{self.refined_hypothesis}\n\nUser Feedback:\n\n{user_feedback}\n\n"
        content += "Please adjust the refined hypothesis based on the user feedback, focusing on photosynthesis-related insights."

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a research assistant specializing in photosynthesis. Adjust the refined hypothesis based on the user feedback. If no user feedback is provided, return the refined hypothesis with the exact wording."},
                {"role": "user", "content": content}
            ],
            temperature=0.7
        )
        self.refined_hypothesis = response.choices[0].message['content'].strip()

        print("\nAdjusted Hypothesis:")
        print(self.refined_hypothesis)

    def run_arxiv_process(self):
        self.generate_search_query()
        self.retrieve_literature()
        self.summarize_literature()
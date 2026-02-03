from abc import ABC, abstractmethod
from utils.literature_retrieval import generate_search_query, retrieve_from_arxiv
import openai
from config.settings import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

class BaseHypothesisAgent(ABC):
    def __init__(self, hypothesis):
        self.hypothesis = hypothesis
        self.search_query = None
        self.literature = None
        self.summary = None
        self.refined_hypothesis = None

    def generate_search_query(self):
        self.search_query = generate_search_query(self.hypothesis)

    def retrieve_literature(self):
        self.literature = retrieve_from_arxiv(self.search_query)

    def summarize_literature(self):
        formatted_lit = '\n'.join([f"{item['title']}: {item['link']}" for item in self.literature])
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in scientific literature summarization. Provide a concise summary of the key findings and insights from the given literature."},
                {"role": "user", "content": f"Here is the list of literature:\n\n{formatted_lit}\n\nPlease summarize the key findings and insights from these papers."}
            ],
            temperature=0.7
        )
        self.summary = response.choices[0].message['content'].strip()

    @abstractmethod
    def refine_hypothesis(self, evaluation_result=None):
        pass
    
    @abstractmethod
    def adjust_hypothesis(self, user_feedback=None):
        pass
    
    @abstractmethod
    def run_arxiv_process(self):
        pass

    def get_refined_hypothesis(self):
        return self.refined_hypothesis
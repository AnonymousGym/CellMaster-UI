# agents/experiment_agent/photosynthesis_experiment_agent.py

from agents.experiment_agent.base_experiment_agent import BaseExperimentAgent
import openai
from config.settings import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

class PhotosynthesisExperimentAgent(BaseExperimentAgent):
    def propose_experiment(self):
        simulation_background = '''
        The photosynthesis simulation allows users to investigate the complex process by which plants convert light energy into chemical energy. By manipulating key variables and observing the effects on photosynthetic outputs, users can develop a deeper understanding of the factors that influence this critical biological pathway.

        The simulation provides a controlled environment to explore the relationships between light, temperature, and other parameters, and the resulting production of glucose, oxygen, and key intermediates. Through experimentation, users can gain insights into the optimization of photosynthesis under various conditions.

        This interactive model serves as a valuable tool for learning about the intricacies of photosynthesis and its role in sustaining life on Earth. By engaging with the simulation, users can appreciate the elegant complexity of this process and its significance in global ecosystems.
        '''

        simulation_input_format = '''
        {
            "distance_from_light": [0, 150],
            "initial_ph_of_stroma": [0, 14],
            "added_NADPH": [0, 50],
            "carbon_dioxide": [0, 100],
            "temperature": [0, 100],
            "atrazine": [0, 1]
        }
        '''

        prompt = f'''
        Summary of literature: {self.literature_summary}
        Refined hypothesis: {self.hypothesis}
        Simulation background: {simulation_background}
        Simulation input format: {simulation_input_format}

        Based on the provided summary of literature and the refined hypothesis, propose an experiment using the photosynthesis simulation. The experiment should aim to test the refined hypothesis and provide insights into the factors influencing photosynthesis.

        Generate the experiment proposal strictly in the form of the simulation input format (json format), including the curly brackets. For each parameter, provide the values to be used in the simulation. The number of values for different parameters should be strictly the same.

        - If a parameter has not been manipulated from the beginning, use the keyword "default".
        - If a parameter has not been manipulated from the beginning while other parameters are changing, repeat the keyword "default" for each value of the changing parameters.
        - Ensure all instances of the word "default" are enclosed in double quotes ("default").
        - If a parameter has been manipulated previously but stays unchanged while other parameters are changing, repeat its unchanged value for each value of the changing parameters.
        - If multiple values are to be tested for a parameter, list them separated by commas.
        Ensure that the output format has the same number of values for each parameter, with "default" repeated for unchanged parameters.

        return only the experiment proposal in the simulation input format (json format), without any additional information.
        '''

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI trained to design scientific experiments based on hypotheses and background information."},
                {"role": "user", "content": prompt}
            ],
        )

        self.experiment_proposal = response.choices[0].message['content'].strip()
        
    def adjust_experiment(self, user_feedback):
        content = f"Experiment Proposal:\n\n{self.experiment_proposal}\n\nUser Feedback:\n\n{user_feedback}\n\n"
        content += "Please adjust the experiment proposal based on the user feedback, focusing on photosynthesis-related insights."

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a research assistant specializing in photosynthesis. Adjust the experiment proposal based on the user feedback."},
                {"role": "user", "content": content}
            ],
            temperature=0.7
        )
        self.experiment_proposal = response.choices[0].message['content'].strip()

        print("\nAdjusted Experiment Proposal:")
        print(self.experiment_proposal)
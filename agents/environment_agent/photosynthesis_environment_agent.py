# agents/environment_agent/photosynthesis_environment_agent.py

from agents.environment_agent.base_environment_agent import BaseEnvironmentAgent
from config.settings import SIMULATION_ENVIRONMENTS, OUTPUT_DIR
import json
import os

class PhotosynthesisEnvironmentAgent(BaseEnvironmentAgent):
    def __init__(self, input_dir, output_dir):
        super().__init__(simulation_environment='photosynthesis', input_dir=input_dir, output_dir=output_dir)
    
    def create_interaction(self):
        from utils.web_interaction.photosynthesis_interaction import PhotosynthesisInteraction
        return PhotosynthesisInteraction(SIMULATION_ENVIRONMENTS['photosynthesis']['url'])

    def extract_experiment_values(self, experiment_proposal):
        parameter_values = json.loads(experiment_proposal)
        
        distance_from_light = parameter_values.get('distance_from_light', ['default'])
        initial_ph_of_stroma = parameter_values.get('initial_ph_of_stroma', ['default'])
        added_NADPH = parameter_values.get('added_NADPH', ['default'])
        carbon_dioxide = parameter_values.get('carbon_dioxide', ['default'])
        temperature = parameter_values.get('temperature', ['default'])
        atrazine = parameter_values.get('atrazine', ['default'])
        
        experiment_values = {
            'distance_from_light': distance_from_light,
            'initial_ph_of_stroma': initial_ph_of_stroma,
            'added_NADPH': added_NADPH,
            'carbon_dioxide': carbon_dioxide,
            'temperature': temperature,
            'atrazine': atrazine
        }
        
        return experiment_values
    
    def run_experiment(self, experiment_proposal):
        experiment_values = self.extract_experiment_values(experiment_proposal)
        interaction = self.create_interaction()
        interaction.run_experiment(experiment_values)
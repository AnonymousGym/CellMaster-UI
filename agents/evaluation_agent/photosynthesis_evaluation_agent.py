import os
import openai
import base64
import requests
from config.settings import OPENAI_API_KEY
from agents.evaluation_agent.base_evaluation_agent import BaseEvaluationAgent

openai.api_key = OPENAI_API_KEY

class PhotosynthesisEvaluationAgent(BaseEvaluationAgent):
    def __init__(self, hypothesis, output_dir):
        super().__init__(hypothesis, output_dir)

    def evaluate(self):
        image_files = [f for f in os.listdir(self.output_dir) if f.startswith('chart_') and f.endswith('.png')]
        image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort images by index

        messages = [
            {"role": "system", "content": "You are a scientist analyzing charts for trends, patterns, or relationships related to photosynthesis."},
            {"role": "user", "content": "Here are examples of photosynthesis chart analyses:\n"
                    "1. Chart showing light intensity vs. photosynthetic rate: As light intensity increases, the photosynthetic rate increases until it reaches a plateau, indicating light saturation.\n"
                    "2. Chart comparing photosynthetic rates under different CO2 concentrations: Higher CO2 concentrations lead to increased photosynthetic rates, demonstrating the importance of CO2 availability for photosynthesis."}
        ]

        for image_file in image_files:
            with open(os.path.join(self.output_dir, image_file), "rb") as f:
                image_data = f.read()
                base64_image = base64.b64encode(image_data).decode("utf-8")
            
            messages.append(
                {"role": "user", "content": [{"type": "image_url", "image_url": f"data:image/png;base64,{base64_image}"}]}
            )

        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=messages,
            max_tokens=300
        )

        return response.choices[0].message['content'].strip()
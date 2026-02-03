# utils/web_interaction/photosynthesis_interaction.py

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from utils.web_interaction.base_interaction import BaseInteraction
import time
from config.settings import OUTPUT_DIR

class PhotosynthesisInteraction(BaseInteraction):
    def run_experiment(self, experiment_values):
        distance_from_light = experiment_values.get('distance_from_light', ['default'])
        initial_ph_of_stroma = experiment_values.get('initial_ph_of_stroma', ['default'])
        added_NADPH = experiment_values.get('added_NADPH', ['default'])
        carbon_dioxide = experiment_values.get('carbon_dioxide', ['default'])
        temperature = experiment_values.get('temperature', ['default'])
        atrazine = experiment_values.get('atrazine', ['default'])

        num_simulations = max(
            len(distance_from_light),
            len(initial_ph_of_stroma),
            len(added_NADPH),
            len(carbon_dioxide),
            len(temperature),
            len(atrazine)
        )

        for i in range(num_simulations):
            self.driver.get(self.simulation_url)
            if i < len(distance_from_light):
                self.find_and_interact_with_element("#interface_pg3_numeric_input_2", distance_from_light[i])
                print(f"Distance from light: {distance_from_light[i]}")
            if i < len(initial_ph_of_stroma):
                self.find_and_interact_with_element("#interface_pg3_numeric_input_4", initial_ph_of_stroma[i])
                print(f"Initial pH of stroma: {initial_ph_of_stroma[i]}")
            if i < len(added_NADPH):
                self.find_and_interact_with_element("#interface_pg3_numeric_input_5", added_NADPH[i])
                print(f"Added NADPH: {added_NADPH[i]}")
            if i < len(carbon_dioxide):
                self.find_and_interact_with_element("#interface_pg3_numeric_input_6", carbon_dioxide[i])
                print(f"Carbon dioxide: {carbon_dioxide[i]}")
            if i < len(temperature):
                self.find_and_interact_with_element("#interface_pg3_numeric_input_7", temperature[i])
                print(f"Temperature: {temperature[i]}")
            if i < len(atrazine):
                self.find_and_interact_with_element("#interface_pg3_numeric_input_1", atrazine[i])
                print(f"Atrazine: {atrazine[i]}")
            self.click_run_button()
            print('--' * 20)
            self.capture_chart(i)

    def find_and_interact_with_element(self, css_selector, value):
        element = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, css_selector))
        )
        self.driver.execute_script(f"""
            var input = document.querySelector("{css_selector}").shadowRoot.querySelector("#input");
            input.value = arguments[0];
            input.dispatchEvent(new Event('input'));
            input.dispatchEvent(new Event('change'));
        """, value)

    def click_run_button(self):
        button_element = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#interface_pg3_button_5"))
        )
        self.driver.execute_script("""
            var button = document.querySelector("#interface_pg3_button_5").shadowRoot.querySelector("#iconContainer");
            button.click();
        """)
        print("Run experiment")

    def capture_chart(self, index):
        sim_chart_element = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#interface_pg3_graph_1"))
        )
        sim_chart_element = self.driver.execute_script("""
            return document.querySelector("#interface_pg3_graph_1").shadowRoot.querySelector("#chart")
        """)
        time.sleep(5)
        chart_name = f"{OUTPUT_DIR['photosynthesis']}/chart_{index}.png"
        sim_chart_element.screenshot(chart_name)
        print(f"Chart {index} captured")
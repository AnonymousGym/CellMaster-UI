# utils/web_interaction/base_interaction.py

from abc import ABC, abstractmethod
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from config.settings import CHROMEDRIVER_PATH, HEADLESS_MODE

class BaseInteraction(ABC):
    def __init__(self, simulation_url):
        self.simulation_url = simulation_url
        self.driver = self.setup_driver()

    def setup_driver(self):
        chrome_options = Options()
        if HEADLESS_MODE:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')

        chrome_service = Service(executable_path=CHROMEDRIVER_PATH)
        driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
        return driver

    @abstractmethod
    def run_experiment(self, experiment_values):
        pass

    @abstractmethod
    def click_run_button(self):
        pass

    @abstractmethod
    def capture_chart(self, index):
        pass

    def close(self):
        self.driver.quit()
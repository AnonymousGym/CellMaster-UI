# config/settings.py

# Simulation environment settings
SIMULATION_ENVIRONMENTS = {
    'photosynthesis': {
        'url': 'https://example.com/photosynthesis-simulator',
    }
}

# Chart output directory settings
OUTPUT_DIR = {
    'photosynthesis': 'data/photosynthesis/output',
    'liver': 'data/liver/output'
}

# OpenAI API settings
OPENAI_API_KEY = 'your-openai-api-key-here'  # Replace with your actual OpenAI API keys

# Literature retrieval settings
SEARCH_ENGINE = 'arxiv'  # Options: 'google_scholar', 'arxiv'
MAX_RESULTS = 5

# Selenium settings
CHROMEDRIVER_PATH = "/usr/local/bin/chromedriver"  # Replace with the path to your ChromeDriver executable
HEADLESS_MODE = True
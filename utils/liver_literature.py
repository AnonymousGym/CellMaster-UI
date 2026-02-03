import requests
from bs4 import BeautifulSoup
import openai
from config.settings import OPENAI_API_KEY, SEARCH_ENGINE, MAX_RESULTS
openai.api_key = OPENAI_API_KEY

def generate_search_query(hypothesis):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI trained to assist with scientific research."},
            {"role": "user", "content": f"Generate a search query for relevant literature based on the following hypothesis: {hypothesis}. Return only the query without any quotes and other text"}
        ],
    )
    search_query = response.choices[0].message['content'].strip()
    print('Search query: ', search_query)
    return search_query

def retrieve_from_google_scholar(search_query):
    print('retrieving from google scholar')
    search_query = search_query.strip('"')
    base_url = "https://scholar.google.com/scholar?q="
    url = base_url + search_query.replace(" ", "+")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    literature = []
    for result in soup.select(".gs_rt a")[:MAX_RESULTS]:
        title = result.text
        link = result["href"]
        literature.append({"title": title, "link": link})

    return literature

def summarize_literature(literature,interest_in_paper):
    formatted_literature = '\n'.join([f"{item['title']}: {item['link']}" for item in literature])
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI trained to summarize scientific literature."},
            {"role": "user", "content": f"The following is a list of literature with their titles and links. {formatted_literature} Visit the links and answer the question {interest_in_paper} for each literature:\n\n"}
        ],
    )
    summary = response.choices[0].message['content'].strip()
    return summary

def ask_specific_question(related_paper_description,interest_in_paper,query=None):
    if query:
        literature = retrieve_from_google_scholar(query)
        if len(literature) == 0:
            print("Query not working, change one or abandon query")
            return
        summary = summarize_literature(literature,interest_in_paper)
    else:
        literature = []
        count = 0
        while len(literature) == 0:
            count += 1
            if count == 10:
                print("try other input please")
                return
            search_query = generate_search_query(related_paper_description)
            literature = retrieve_from_google_scholar(search_query)
        summary = summarize_literature(literature,interest_in_paper)
    return literature,summary
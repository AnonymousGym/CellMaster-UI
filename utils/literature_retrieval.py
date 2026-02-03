# utils/literature_retrieval.py

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

def retrieve_literature(search_query):
    if SEARCH_ENGINE == 'arxiv':
        return retrieve_from_arxiv(search_query)
    else:
        raise ValueError(f"Unsupported search engine: {SEARCH_ENGINE}")

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

def retrieve_from_arxiv(search_query, max_results=10):
    print('retrieving from arxiv')
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
    }

    response = requests.get(base_url, params=params)
    response.raise_for_status()

    literature = []
    entries = response.text.split("<entry>")[1:]

    for entry in entries:
        title_start = entry.find("<title>") + len("<title>")
        title_end = entry.find("</title>")
        title = entry[title_start:title_end]

        link_start = entry.find("<link href=") + len("<link href=")
        link_end = entry.find("/>", link_start) - 1
        link = entry[link_start:link_end].strip('"')

        literature.append({"title": title, "link": link})
        
    print('----------------------------------------------------------------------------------')
    print('List of literature:')
    print(literature)
    print('----------------------------------------------------------------------------------')

    return literature

def summarize_literature(literature):
    formatted_literature = '\n'.join([f"{item['title']}: {item['link']}" for item in literature])
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI trained to summarize scientific literature."},
            {"role": "user", "content": f"The following is a list of literature with their titles and links. Visit the linkds and generate a summary for each literature:\n\n{formatted_literature}"}
        ],
    )
    summary = response.choices[0].message['content'].strip()
    return summary
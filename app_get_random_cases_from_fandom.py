import requests
import json
from bs4 import BeautifulSoup
import random
import csv

WIKI_API_URL = "https://casebrief.fandom.com/api.php"

cases = [
    "Christie v York", "R v Miller", "Abdo v Abdo", "Andrews v Grand & Toy Alberta Ltd.",
    "Anns v Merton London Borough Council", "Appleby v Erie Tobacco Co.", "Arnold v Teno",
    "Asylum Case (Colombia v Peru)", "Athey v Leonati", "Baker v Canada (Minister of Citizenship and Immigration)"
]

def get_category_pages(category, limit=2):
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": limit,
        "format": "json"
    }
    response = requests.get(WIKI_API_URL, params=params)
    data = response.json()
    return data['query']['categorymembers']

def get_page_content(pageTitle):
    params = {
        "action": "parse",
        "prop": "text",
        "page": pageTitle,
        "format": "json"
    }
    response = requests.get(WIKI_API_URL, params=params)
    data = response.json()
    soup = BeautifulSoup(data['parse']['text']['*'], 'html.parser')

    details = {
        "Title": pageTitle,
        "Citation": "",
        "Appellant": "",
        "Respondent": "",
        "Year": "",
        "Court": "",
        "Judges": "",
        "Country": "",
        "Province": "",
        "Area of Law": "",
        "Issue": "",
        "Facts": "",
        "Decision": "",
        "Reasons": "",
        "Ratio": ""
    }

    infobox = soup.find("aside", class_="portable-infobox")
    if infobox:
        for item in infobox.find_all("div", class_="pi-item"):
            label = item.find("h3", class_="pi-data-label")
            value = item.find("div", class_="pi-data-value")
            if label and value:
                label_text = label.get_text(strip=True)
                value_text = value.get_text(strip=True)
                if label_text in details:
                    details[label_text] = value_text

    content_sections = ["Facts", "Issue", "Decision", "Reasons", "Ratio"]
    current_section = None
    section_texts = {section: "" for section in content_sections}

    for tag in soup.find_all(['h2', 'p', 'ul', 'ol']):
        if tag.name == 'h2':
            section_title = tag.get_text(strip=True).split('[')[0]
            if section_title in section_texts:
                current_section = section_title
        elif current_section:
            section_texts[current_section] += tag.get_text(separator='\n', strip=True) + "\n"

    for section in section_texts:
        details[section] = section_texts[section].strip()

    return details

def main():
    random_cases = get_category_pages("Supreme_Court_of_Canada_cases", limit=100)
    selected_cases = random.sample(random_cases, 50)
    all_details = []

    for case in selected_cases:
        title = case['title']
        try:
            content = get_page_content(title)
            all_details.append(content)
        except Exception as e:
            print(f"Error processing {title}: {e}")

    keys = all_details[0].keys() if all_details else []
    with open('random_cases.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_details)

if __name__ == "__main__":
    main()

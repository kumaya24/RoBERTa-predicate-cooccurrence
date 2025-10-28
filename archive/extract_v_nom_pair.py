import requests
import random
import re
import time
import os
from bs4 import BeautifulSoup
from typing import List, Optional

WIKI_BASE_URL = "https://en.wiktionary.org/wiki/"
HEADERS = {"User-Agent": "WiktionaryNominalizationBot/1.0 (jacquihardenuc21@gmail.com)"}
INPUT_VERB_FILE = "all_verified_verbs.txt"
OUTPUT_FILE = "verb_nominalization_pairs.txt"
MAX_VERIFY_CANDIDATES_PER_VERB = 20 
REQUEST_TIMEOUT = 10 

def load_verified_verbs(filename: str) -> List[str]:
    if not os.path.exists(filename):
        print(f"Error: Input file '{filename}' not found. Please run fetch_verbs.py first.")
        return []
    
    with open(filename, 'r', encoding='utf-8') as f:
        return list({line.strip().lower() for line in f if line.strip()})

def fetch_page_content(word: str, session: requests.Session) -> Optional[str]:
    url = WIKI_BASE_URL + word
    try:
        response = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status() 
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching page for '{word}': {e}")
        return None

def extract_derived_terms(html_content: str, verb_initial: str) -> List[str]:
    soup = BeautifulSoup(html_content, 'html.parser')
    candidates = []

    verb_span = soup.find('span', id='Verb')
    if not verb_span:
        return [] 

    verb_header = verb_span.find_parent(['h3', 'h4'])
    if not verb_header:
        return []

    derived_section = None
    related_terms = None
    current_element = verb_header.next_sibling
    
    while current_element:
        if current_element.name in ['h3', 'h4']:
            break 

        derived_span = current_element.find('span', id=re.compile(r'Related_terms'))
        related_span = current_element.find('span', id=re.compile(r'Related_terms'))
        if derived_span:
            derived_section = derived_span.find_parent().find_next(['ul', 'ol', 'div'])
            break 

        if related_span:
            related_terms = related_span.find_parent().find_next(['ul', 'ol', 'div'])
            break
        
        current_element = current_element.next_sibling
        
    if not derived_section:
        derived_section = soup.find('div', {'data-toggle-category': 'derived terms'})

    if not related_terms:
        related_terms = soup.find('div', {'data-toggle-category': 'related terms'})


    if derived_section:
        term_links = derived_section.find_all('a', title=True)
        
        for link in term_links:
            #if link.get('title') == "Category:English compound terms":
            #    continue

            candidate_word = link.get('title', link.text).split('#')[0] 
            
            candidate_word = re.sub(r'\s*\(.*\)\s*', '', candidate_word).strip()
            
            # if not candidate_word or candidate_word.lower() == 'appendix:english_suffixes':
            #     continue

            if " " in candidate_word or "-" in candidate_word:
                continue

            if candidate_word.lower().startswith(verb_initial):
                if candidate_word.lower() != verb_initial:
                    candidates.append(candidate_word)

    return list(set(candidates)) 

def verify_is_noun(word: str, session: requests.Session) -> bool:
    html_content = fetch_page_content(word, session)
    if not html_content:
        return False
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 1. Find the English section header
    english_span = soup.find('span', id='English')
    if not english_span:
        return False # Word doesn't have an English section
    
    # 2. Get the main content block for English
    # The header is typically H2; the content follows until the next H2.
    english_header = english_span.find_parent('h2')
    if not english_header:
        # Fallback if structure is unusual, but should find H2
        return False
        
    # 3. Iterate through elements *after* the English header
    current_element = english_header.next_sibling
    
    while current_element:
        # Stop at the next language section (usually another H2)
        if current_element.name == 'h2':
            break 
            
        # Check for the Noun header span within the current element's scope
        # Note: 'Noun' is often an H3 or H4 header within the English H2 section
        noun_span = current_element.find('span', id='Noun')
        if noun_span:
            return True # Found the Noun section within the English language block
        
        current_element = current_element.next_sibling
        
    return False # Noun section not found within the English language block

def run_extraction_pipeline():
    verified_verbs = load_verified_verbs(INPUT_VERB_FILE)
    if not verified_verbs:
        print("Exiting pipeline due to no input verbs.")
        return

    session = requests.Session()
    nominalization_pairs = []
     
    random.shuffle(verified_verbs)

    for i, verb in enumerate(verified_verbs):
        if i % 50 == 0 and i > 0:
            print(f"--- Processed {i}/{len(verified_verbs)} verbs. Found {len(nominalization_pairs)} pairs. ---")
            time.sleep(1) 
            
        verb_initial = verb[0].lower()
        verb_page_content = fetch_page_content(verb, session)
        if not verb_page_content:
            continue
            
        found_nouns = []
        content_lower = verb_page_content.lower()
        
        if 'id="noun"'.lower() in content_lower and 'id="english"'.lower() in content_lower:
             found_nouns.append(verb)

        candidates = extract_derived_terms(verb_page_content, verb_initial)
        
        if not candidates and not found_nouns:
            continue
            
        random.shuffle(candidates)
        candidates_to_verify = candidates[:MAX_VERIFY_CANDIDATES_PER_VERB]

        for candidate in candidates_to_verify:
            time.sleep(0.1) 
            if verify_is_noun(candidate, session):
                found_nouns.append(candidate)
        
        if found_nouns:
            unique_sorted_nouns = sorted(list(set(found_nouns)))
            nominalization_pairs.append(f"{verb} | {', '.join(unique_sorted_nouns)}")
            
    print("done")
if __name__ == "__main__":
    run_extraction_pipeline()
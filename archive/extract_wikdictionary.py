import requests
import random
import re
import string
import time

# --- Verification Function ---

def verify_word_is_verb(word, session, headers):
    """
    Fetches the Wiktionary page for a word and checks for an HTML marker 
    that confirms it is categorized as an English verb (Part-of-Speech = Verb).
    
    A successful check looks for the anchor ID usually associated with the 
    English verb section, such as an h3 header containing the text 'Verb' 
    under the main 'English' language section.
    """
    wiki_url = f"https://en.wiktionary.org/wiki/{word}"
    
    try:
        response = session.get(wiki_url, headers=headers, timeout=5)
        response.raise_for_status()
        html_content = response.text
        if '<span id="Verb">Verb</span>' in html_content:
            return True
        if 'id="English"'.lower() in html_content.lower():
            if 'id="Verb"'.lower() in html_content.lower():
                return True
            
    except requests.exceptions.RequestException as e:
        # This handles network errors, timeouts, or 404s (word not found/page error)
        # print(f"Verification failed for {word}: {e}")
        pass
        
    return False

def get_verbs_from_wiktionary(cmlimit=500):
    """
    Retrieves a list of single-word titles from Wiktionary's 'English verbs' category.
    Handles pagination to get a larger list of candidates.
    """
    api_url = "https://en.wiktionary.org/w/api.php"
    candidate_list = []
    
    # We use a User-Agent to identify our script, which is good practice.
    headers = {
        "User-Agent": "MyBot/1.0 (jacquihardenuc21@gmail.com)"
    }
    
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": "Category:English verbs",
        "cmlimit": cmlimit,
        "cmprop": "title"
    }

    
    while True:
        try:
            response = requests.get(api_url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"HTTP error during API fetch: {e}")
            return []
        except requests.exceptions.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Response text: {response.text[:500]}...")
            return []

        if "query" not in data or "categorymembers" not in data["query"]:
            print("Unexpected API response structure.")
            return []
        
        for item in data['query']['categorymembers']:
            title = item['title']
            if " " not in title and "-" not in title: 
                candidate_list.append(title.lower())
        
        if 'continue' in data:
            params.update(data['continue'])

        else:
            break

    return candidate_list

def main():
    candidate_verbs = get_verbs_from_wiktionary(cmlimit=500)
    
    if not candidate_verbs:
        print("Failed to retrieve any candidates. Exiting.")
        return

    # --- STEP 3: Verify POS using page source check ---
    
    # We will only verify a maximum of 1000 words due to time constraints
    # and to limit repeated network requests.
    MAX_VERIFY = 2000 
    random.shuffle(candidate_verbs)
    verbs_to_verify = candidate_verbs[:MAX_VERIFY]
    
    verified_verbs = []
    verification_session = requests.Session()
    
    # Use a common header for verification requests
    headers = {"User-Agent": "WiktionaryPOSVerifier/1.0 (jacquihardenuc21@gmail.com)"}

    print(f"\nStarting verification of {len(verbs_to_verify)} candidate words (checking page source for 'Verb' tag)...")
    
    for i, word in enumerate(verbs_to_verify):
        if i % 100 == 0 and i > 0:
            print(f"Verified {i} words so far...")
            # Pause to respect rate limits if performing many checks
            # time.sleep(2) 
        
        if verify_word_is_verb(word, verification_session, headers):
            verified_verbs.append(word)

    # --- STEP 4: Finalize and save results ---
    
    if not verified_verbs:
        print("\nVerification process completed, but no words were confirmed as verbs.")
        return
    
    random.shuffle(verified_verbs)
    sample_size = min(1000, len(verified_verbs))
    random_sample = verified_verbs[:sample_size]
    
    print(f"\nVerification complete.")
    print(f"Total verified single-word verbs: {len(verified_verbs)}")
    
    # Print random sample
    print(f"\nRandom sample of {sample_size} verified verbs:")
    for verb in random_sample:
        print(verb)
        
    # Save results to files
    with open("all_verified_verbs.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(verified_verbs))
    
    with open("random_single_word_verbs_verified.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(random_sample))
    
    print("\nFiles have been saved:")
    print("- all_verified_verbs.txt (contains all confirmed verbs)")
    print("- random_single_word_verbs_verified.txt (contains the random sample)")

if __name__ == "__main__":
    main()

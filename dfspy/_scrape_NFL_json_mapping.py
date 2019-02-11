"""
Get column names for NFL.com json scrapes.
"""

import json
from time import sleep

import requests


def main():
    # Disable warnings for unverified requests.
    requests.packages.urllib3.disable_warnings()
    
    url = 'https://api.fantasy.nfl.com/v1/game/stats?format=json'
    # Scrape data, attempt 5 times before passing.
    attempts = 5
    for attempt in range(attempts):
        try:
            r = requests.get(url, verify=False, timeout=5)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            sleep(1)
            r = None
            continue
        else:
            break
        
    if (r is not None and
        r.status_code == 200 and
        'stats' in r.json()):
        # Succesfull scrape; get column names.
        col_names = {sub_dict['id']: sub_dict['shortName'] \
                     for sub_dict in r.json()['stats']}
        
        # Output to json file.
        with open('../data/.NFL_json_columns.json', 'w') as fid:
            json.dump(col_names, fid, indent=4, sort_keys=True)

if __name__ == '__main__':
    main()

# The dates we have for each company
# primary:
    # AMC: 2013-12-18
    # BB: 1999-11-01
    # GME: 2002-02-13
# secondary:
    # BBBY: 1999-11-01
    # KOSS: 1999-11-01
    # PLUG: 1999-11-01
    # TSLA: 2010-06-29

import requests
import json

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
_url1 = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&outputsize=full&symbol="
_symbol = "BBBY" # this should be dynamic
_url2 = "&apikey=6KX0TXEWMWVOT1NK"

_fullURL = _url1 + _symbol + _url2
request = requests.get(_fullURL)
data = request.json()

with open(('alphaVanData_' + _symbol + '_full_daily.json'), 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)


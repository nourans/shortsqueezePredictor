import requests
import json

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
_url1 = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol="
_symbol = "PLUG" # this should be dynamic
_url2 = "&interval=5min&month=2009-01&outputsize=full&apikey=6KX0TXEWMWVOT1NK"

_fullURL = _url1 + _symbol + _url2
request = requests.get(_fullURL)
data = request.json()

with open(('alphaVanData_'+_symbol+'.json'), 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
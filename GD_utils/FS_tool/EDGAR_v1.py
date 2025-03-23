import pandas as pd
import requests

import requests

headers = {
    "User-Agent": "MyFinanceApp/1.0 (your_email@example.com)"
}

url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"  # 예: AAPL(애플) CIK
response = requests.get(url, headers=headers)
print(response.status_code)
print(response.text)  # 혹시 에러메시지가 있는지 확인



headers = {
    "User-Agent": "FinancialResearch/1.0 (gdpresent@naver.com)"
}
url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"  # Apple Inc.'s Company Facts
response = requests.get(url, headers=headers)
data = response.json()

# Get the JSON mapping of tickers to CIKs from SEC (no auth needed, but include User-Agent)
map_url = "https://www.sec.gov/files/company_tickers.json"
res = requests.get(map_url, headers=headers)
ticker_map = res.json()
das
# The JSON is indexed by number, so build a dictionary for quick lookup:
cik_lookup = {}
for item in ticker_map.values():
    ticker = item['ticker']
    cik_str = str(item['cik_str']).zfill(10)  # zero-pad to 10 digits
    cik_lookup[ticker.upper()] = cik_str

print(cik_lookup.get("AAPL"))  # e.g., should print "0000320193"
import pandas as pd
import requests

def convert_to_dataframe(data):
    df = pd.DataFrame(data)
    return df

def fetch_data_from_api(url):
    response = requests.get(url)
    if response.status_code == 200:
        return convert_to_dataframe(response.json()['data'])
    else:
        print("Failed to fetch data:", response.status_code)
        return {}


def fetch_stock_data(ticker):
    url = f"http://0.0.0.0:8080/stocks/{ticker}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

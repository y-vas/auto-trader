import requests
from dotenv import dotenv_values, find_dotenv

# api url
_url  = 'https://api.ymvas.com'

# parse the .env values 
config = dotenv_values(find_dotenv())

def api( path ):
    data = requests.get(
        _url + path,
        headers = { "API_PASS" : config['YMVAS_API'] }
    )

    
    return data.json()

# function for obtaining all the avaliable instruments on the plataform
def get_instruments():
    instruments_dict = api("/stocks/instruments")['data']

    return [ instruments_dict[x] for x in instruments_dict ]

# function for obtaining hitorical data of the given instrument
def get_candles( instrument_id ):
    return api(
        "/stocks/instruments/candles"
        f"?instrument={instrument_id}"
    )['data']

# function for normalized data of certain instrument
# heavy function
def get_dataset( instrument, year , page = 0):
    
    return api(
        "/stocks/datasets/related_stocks"  
        f"?year={year}"
        f"&sequences=200"
        f"&page={page}"
        f"&target={instrument}"
    )
     
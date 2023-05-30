from apis.ymvas import get_instruments, get_candles, get_dataset


# returns a list of all the avaliable instruments
instruments = get_instruments()
print(instruments[0]) # shows the first instrument



# hitorical data of the given instrument
aud_cad_candles = get_candles(1)
# you can also use this:
# candles = get_candles(instruments[0].id)
# candles are the raw data that is requiered to be modified (normalized)
# in order to be managable by the ML models in /models/ folder



dataset = get_dataset('EUR_USD',2022,0)
# datase is the already normalized data that can be easly added to the 
# learning model and get insights about something






# a partir de aqui no he tingut mes temps per documentar ... 
# la questio es montar preparar les dades perque entrin en 
# els models sequencials i es puguis obtenir informacio relevant 
# ajuntare mes informacio pero de moment aixo es tot

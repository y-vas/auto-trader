import talib
from talib import *

def ta_lib_all( df, close, open, high, low, volume ):

    talibf = get_function_groups()
    for candle in talibf['Pattern Recognition']:
        df[candle] = getattr(talib, candle)( open, high, low, close )
        df[candle] /= 200

    df['RSI'] = RSI(close, timeperiod = 14)
    df['RSI'] /= 100

    df['BBU'], df['BBM'], df['BBL'] = BBANDS( close , timeperiod = 5, nbdevup=2, nbdevdn=2, matype = 0 )
    df['BBU'] = ((df['BBU']-close) / df['BBU'] )
    df['BBM'] = ((df['BBM']-close) / df['BBM'] )
    df['BBL'] = ((df['BBL']-close) / df['BBL'] )

    df['HT_TRENDLINE'] = HT_TRENDLINE( close )
    df['HT_TRENDLINE'] = ((df['HT_TRENDLINE']-close) / df['HT_TRENDLINE'] )
    df['HT_TRENDLINE'] = df['HT_TRENDLINE'].fillna( 0 )
    df['HT_TRENDMODE'] = HT_TRENDMODE(close)

    df['MIN'], df['MAX'] = MINMAX(close, timeperiod = 30 )
    df['MIN'] = df['MIN'] == close
    df['MAX'] = df['MAX'] == close

    df['MIN'] = list(map( lambda d: int(d) , df['MIN'] ))
    df['MAX'] = list(map( lambda d: int(d) , df['MAX'] ))

    df['MIDPOINT'] = MIDPOINT(close ,      timeperiod = 14 )
    df['MIDPOINT'] = ((df['MIDPOINT']-close) / df['MIDPOINT'] )

    df['SMA'] = SMA( close , timeperiod = 30 )
    df['SMA'] = ((df['SMA']-close) /df['SMA'])

    df['T3']   = T3( close, timeperiod= 5, vfactor=0)
    df['T3'] = ((df['T3']-close) /df['T3'])

    df['TEMA'] = TEMA( close, timeperiod = 30 )
    df['TEMA'] = ((df['TEMA']-close) /df['TEMA'])

    df['TRIMA']= TRIMA(close, timeperiod=30)
    df['TRIMA'] = ((df['TRIMA']-close) /df['TRIMA'])

    df['WMA']  = WMA(  close, timeperiod=30)
    df['WMA'] = ((df['WMA']-close) /df['WMA'])

    df['MACD'], df['MACDSIGLAN'], df['MACDHIST'] = MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] /= 100
    df['MACDSIGLAN'] /= 100
    df['MACDHIST'] /= 100

    df['ADX'] = ADX(high, low, close, timeperiod=14) / 100
    df['ADXR'] = ADXR(high, low, close, timeperiod=14) / 100
    df['APO'] = APO(close, fastperiod=12, slowperiod=26, matype=0) / 100

    df['AROONOSC'] = AROONOSC(high, low, timeperiod=14) / 100
    df['BOP'] = BOP(open, high, low, close)

    df['CCI'] = CCI(high, low, close, timeperiod=14) / 100
    df['CMO'] = CMO(close, timeperiod = 14 ) / 100
    df['DX'] = DX(high, low, close, timeperiod=14) / 100

    df['mama'], df['fama'] = MAMA(close)
    df['mama'] = ((df['mama']-close) /df['mama'])
    df['fama'] = ((df['fama']-close) /df['fama'])

    df['aroondown'], df['aroonup'] = AROON(high, low, timeperiod=14)
    df['aroondown'] /= 100
    df['aroonup'] /= 100

    df['MFI'] = MFI( high, low, close, volume, timeperiod = 14 ) / 100

    df['MINUS_DI'] = MINUS_DI(high, low, close, timeperiod=14) / 100
    df['MINUS_DM'] = MINUS_DM(high, low, timeperiod=14) / 100
    df['MOM'] = MOM(close, timeperiod=10) / 100
    df['PLUS_DI'] = PLUS_DI(high, low, close, timeperiod=14) /100
    df['PLUS_DM'] = PLUS_DM(high, low, timeperiod=14) /100

    df['PPO']  = PPO(  close , fastperiod = 12 , slowperiod=26, matype=0)
    df['ROC']  = ROC(  close , timeperiod = 10 )
    df['ROCP'] = ROCP( close , timeperiod = 10 )
    df['ROCR'] = ROCR( close , timeperiod = 10 )

    df['slowk'], df['slowd'] = STOCH( high  , low, close, fastk_period = 5, slowk_period  = 3, slowk_matype = 0, slowd_period=3, slowd_matype=0)
    df['slowk'] /= 100
    df['slowd'] /= 100

    df['fastk'], df['fastd'] = STOCHF( high  , low, close, fastk_period = 5, fastd_period  = 3, fastd_matype = 0 )
    df['fastk'] /= 100
    df['fastd'] /= 100

    df['macdE'], df['macdEsignal'], df['macdEhist'] = MACDEXT(
        close,
        fastperiod=12,
        fastmatype=0,
        slowperiod=26,
        slowmatype=0,
        signalperiod=9,
        signalmatype=0
    )

    df['macdE'       ] /= 100
    df['macdEsignal' ] /= 100
    df['macdEhist'   ] /= 100

    df['macdF'], df['macdFsignal'], df['macdFhist'] = MACDFIX(close, signalperiod=9)
    df['macdF'       ] /= 100
    df['macdFsignal' ] /= 100
    df['macdFhist'   ] /= 100

    df['TRIX'  ] = TRIX(close, timeperiod=30)
    df['ULTOSC'] = ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28) / 100
    df['WILLR' ] = WILLR(high, low, close, timeperiod=14) / 100

    df['ATR'] = ATR(high, low, close, timeperiod=14) / 100
    df['NATR'] = NATR(high, low, close, timeperiod=14)
    df['TRANGE'] = TRANGE(high, low, close) / 100

    df['HT_DCPERIOD'] = HT_DCPERIOD(close) / 100
    df['HT_DCPHASE'] = HT_DCPHASE(close) / 100

    df['DEMA'] = DEMA(close, timeperiod = 30 )
    df['EMA']  = EMA(close,  timeperiod = 30 )
    df['KAMA'] = KAMA(close, timeperiod = 30 )
    df['MA']   = MA(close,   timeperiod = 30, matype = 0 )

    df['DEMA'] = ((df['DEMA']-close) /df['DEMA'])
    df['EMA'] = ((df['EMA']-close) /df['EMA'])
    df['KAMA'] = ((df['KAMA']-close) /df['KAMA'])
    df['MA'] = ((df['MA']-close) /df['MA'])

from os import makedirs
from datetime import timedelta as td, datetime as dt
from time import sleep, time
from oandapyV20 import API
from collections import deque
from oandapyV20.endpoints.accounts import AccountInstruments as Stocks
from oandapyV20.endpoints import pricing , orders, trades, positions
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.endpoints.pricing import PricingStream
# https://github.com/tensorflow/agents/blob/master/docs/tutorials/1_dqn_tutorial.ipynb

import talib_all

import re, sys, math, json,random, pandas as pd ,numpy as np
from sklearn import preprocessing

from calendar import monthrange
from math import ceil

if not 'nn' in sys.argv:
    import tensorflow as tf
    from tensorflow.python.keras.models import Sequential
    from tensorflow.keras.layers import Dense, GRU, Dropout, LSTM, BatchNormalization
    from tensorflow.keras.models import load_model

import os,math, json, shutil, numpy as np, pandas as pd
from jict import jict

from numpy import savetxt, loadtxt

thispath = os.path.dirname(os.path.realpath(__file__))

class Core:
    mdl = os.path.join(os.environ[ 'models' ], 'm2')
    aid = '101-004-12605728-001'
    api = API('b8714cf099913aaffcc5e13e6679a3f5-53c195784f239a611c0b7c3fafabf3c0')

    ins = dict(api.request(Stocks(aid)))['instruments']
    ins.sort( key = lambda d: d['name'] )

    isd = jict({ x['name']: x for x in ins })
    instrum = jict( thispath + '/source/instruments.json' )['data']
    for x in instrum: isd[x]['id'] = instrum[x]['id'] / 123 # max_instruments

    hfm = [ 'year' ,'month' ,'day' ]
    prod= jict( thispath + '/source/progres.json' )

    def __init__( self ):
        makedirs( os.environ['models'] , exist_ok = True )

        # self.closeall()
        # for x in self.isd:
        #     self.order(x,True,100)

        # exit()

    def specific_train( self , instrument ):
        df = self.repeat( instrument )

        dfX, dfy, last = self.preprocess( df , 64 )

        if dfX.shape[0] == 0:
            return

        self.mdl = os.path.join(os.environ[ 'models' ], instrument )

        model = self.model( dfX.shape[1:] , 4 )

        model.fit(
            dfX, dfy,
            batch_size = 64, epochs = 100 ,
            validation_split = 0.03,
        )

        model.save( self.mdl )


    def train( self ):
        model = None
        for mod, valid, test, last in self.df( split = False ):
            dfX  , dfy  = valid
            dftX , dfty = test

            if dfX.shape[0] == 0:
                continue

            # val_loss starts increasing, val_acc starts decreasing
            # (means model is cramming values not learning)

            # val_loss starts increasing, val_acc also increases.
            # (could be case of overfitting or diverse probability values in
            # cases softmax is used in output layer)

            # val_loss starts decreasing, val_acc starts increasing
            # (Correct, means model build is learning and working fine)

            if model == None:
                model = self.model( dfX.shape[1:] , 4 )

            model.fit(
                dfX, dfy,
                batch_size = 64, epochs = 10 ,
                validation_split = 0.03,
            )

            # Score model //////////////////////////////////////////////////////
            # score = model.evaluate( dftX , dfty , verbose = 0 )

            model.save( self.mdl )

            # tf.keras.backend.clear_session()
            # del model

            # print( last[0] )
            # rY = model.predict( np.array([last[0]]) )
            #
            # print(   rY    )
            # print( last[1] )


    def df(self, split = False ):
        if not isinstance( self.prod['l'] , list ):
            self.prod['l'] = []

        for i, x in enumerate(self.isd):

            # if split == 'load':
            #
            #     if x not in self.prod['l']:
            #         # save
            #         # dfX, dfy, last = self.preprocess( df , 60 )
            #         # x , (dfX, dfy), (None,None) , last
            #         pass
            #     else:
            #         # load
            #         yield df
            #         continue

            if x in self.prod['l']:
                continue

            df = self.repeat( x )

            self.prod['l'].append( x )
            self.prod.save( thispath + '/source/progres.json' )

            if split == 'df':
                yield df

            df['id'] = self.isd[x]['id']

            if split:
                pct = df.index.values[-int(0.05*len(df.index.values))]
                df1 = df[(df.index <  pct)]
                dfX, dfy, _ = self.preprocess( df1 , 64 )

                lasta = df[(df.index >= pct)]

                dftX, dfty, last = self.preprocess( lasta , 64  )
                yield x, (dfX, dfy), ( dftX, dfty ) , last

            dfX, dfy, last = self.preprocess( df , 64 )
            yield x , (dfX, dfy), (None,None) , last

    def repeat(self, name ):
        end = dt.now() - td( hours = 2 )
        return self.candles( name , end )

    def preprocess( self, df , sequences ):
        print(df)

        # scale time
        df['y'] -= 2000
        df['y'] /= 21 # current developing year

        df['d'] /= 31
        df['w'] /= 6
        df['s'] /= 2
        df['t'] /= 4
        df['m'] /= 12

        talib_all.ta_lib_all(df, df['c'], df['o'], df['h'], df['l'], df['v'] )

        pops = [ 'o' ,'c' ,'v' , 'h' , 'l' ,'xtime','index']
        for x in pops:
           df.pop( x )

        # values with .3% increase after 2 days

        df['bshort'] = df['P'].shift(-1).fillna( 0 ) > 0.3
        df['sshort'] = df['P'].shift(-1).fillna( 0 ) <-0.3

        df['blong'] = np.where(( df['U'].shift(-2) >= 2) & df['bshort'] , 1 , 0)
        df['slong'] = np.where(( df['D'].shift(-2) >= 2) & df['sshort'] , 1 , 0)

        df['bshort'] = np.where( df['bshort'] , 1 , 0)
        df['sshort'] = np.where( df['sshort'] , 1 , 0)

        df['P'] /= 100
        df['U'] /= 31
        df['D'] /= 31

        # pd.set_option("display.precision", 3 )
        # # pd.set_option('display.max_rows', None)
        # pd.set_option('display.min_rows', 40)
        # print(df)

        df.dropna( inplace = True )

        sequential_data = []
        prev_days = deque( maxlen = sequences )

        for i in df.values:
            prev_days.append([n for n in i[:-4]])
            if len(prev_days) == sequences:
                sequential_data.append([np.array(prev_days), i[-4:]])

        last = sequential_data[ -1 ]
        sequential_data = sequential_data[:-1]
        random.shuffle( sequential_data )

        buys,sells,mid = [],[],[]
        for seq, target in sequential_data:
            if target[1] == 1:
                sells.append([seq, target])
            elif target[0] == 1:
                buys.append([seq, target])
            else:
                mid.append([seq, target])

        lower = min(len(buys), len(sells), len(mid))

        buys  = buys[  :lower ]
        sells = sells[ :lower ]
        mid   = mid[   :lower ]

        sequential_data = buys + sells + mid
        random.shuffle( sequential_data )

        X, y = [],[]
        for seq, target in sequential_data:
            X.append( seq )
            y.append(target)

        print( len(X) )
        return np.array(X), np.array(y) , last

# CANDLES /--------------------------------------------------------------------/
    def candles(self, name, end = None , size = 5000 , testing = False , granularity = 'D' ):
        format = "%Y-%m-%dT%H:%M:%S.000000000Z"
        strend = end.strftime( format )

        def xtime( tm ):
            return dt.fromtimestamp(dt.timestamp(dt.strptime(tm, "%Y-%m-%dT%H:%M:%S.%f000Z")))

        res = self.api.request( InstrumentsCandles(
            instrument = name , params = {
            'granularity' : granularity ,
            'count'       : size ,
            'to'          : strend,
        }))

        df = pd.DataFrame(  res['candles']   )
        if df.shape[0] == 0:
            return True

        ls = ['open','close','higth','low']

        for x in ls:
            def mid(m): return float(m[x[0]])
            df[x[0]] = list(map(mid,df['mid']))

        df['xtime'] = list( map( xtime, df['time'] ) )

        v = df[ 'volume' ]
        df = df[ ['xtime'] + [ x[0] for x in ls ]]
        df['v'] = v

        for col in self.hfm:
            def mkcol(d):
                return eval( 'd.' + col )
            df[col[0]] = list(map( mkcol, df['xtime'] ))

        df['s' ]= list(map( lambda d: ( d-1 )//6  + 1 , df['m']     ))
        df['t'] = list(map( lambda d: ( d-1 )//3  + 1 , df['m']     ))
        df['w'] = list(map( lambda d: d.weekday()     , df['xtime'] ))
        df['P'] = (( df['c'] - df['o'] ) / df['c'] ) * 100

        ## up and down days ####################################################
        df['U']  = list(map( lambda d: 1 if d > 0 else 0 , df['P'] ))
        df['D']  = list(map( lambda d: 0 if d >=0 else 1 , df['P'] ))

        df['U'] = df['U'] * (df['U'].groupby((df['U'] != df['U'].shift()).cumsum()).cumcount() + 1 )
        df['D'] = df['D'] * (df['D'].groupby((df['D'] != df['D'].shift()).cumsum()).cumcount() + 1 )

        df = df.sort_index( ascending = True )
        df = df.reset_index()

        return df

# ORDERS /---------------------------------------------------------------------/
    def order(self, name, buy = True , waste = 100 ):
        data = self.isd[ name ]

        if len(data) == 0:
            return False

        rv = self.api.request(pricing.PricingInfo(
            accountID = self.aid,
            params    = { "instruments": name }
        ))

        # if you smart you will know what this does
        if not rv[ 'prices' ][0]['tradeable']:
            return False

        metals = float(rv['prices'][0]['quoteHomeConversionFactors']['positiveUnits' if buy else 'negativeUnits'])
        price  = float( rv[ 'prices' ][0]['closeoutBid'] )
        margin = float( self.isd[ name ]['marginRate'] )

        munits = price * metals * margin
        if waste < munits:
            return False

        units = int( waste / munits )

        order = { "order": {
            "units"       : str( units if buy else -units ),
            "instrument"  : name,
            "timeInForce" : "FOK" ,
            "type"        : "MARKET" ,
            "positionFill": "DEFAULT"
        }}

        try:
            self.api.request( orders.OrderCreate( accountID = self.aid , data = order ))
        except Exception as e:
            print( e )
        return True


    def model(self , _in , _out = 2 , epochs = 200 ):

        if os.path.exists( self.mdl ):
            return load_model( self.mdl )

        model = Sequential()

        model.add(LSTM( 162, input_shape=_in, return_sequences=True ))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(LSTM( 128, return_sequences=True ))
        model.add(Dropout(0.1))
        model.add(BatchNormalization())

        # model.add(LSTM( 64 ,return_sequences = True ))
        model.add(LSTM( 64 ))
        model.add(Dropout(0.1))
        model.add(BatchNormalization())

        model.add(Dense( 32 , activation = 'relu' ))
        model.add(Dropout( 0.1 ))

        model.add(Dense( 64 , activation = 'relu' ))
        model.add(Dense( 128, activation = 'relu' ))

        model.add(Dense( units = _out , activation = 'linear' ))

        # model.add( Dense( _out , activation='softmax') )
        # model.summary()

        # learning_rate = 0.0001
        # decay_rate = learning_rate / epochs
        # momentum = 0.8

        # optimizer = tf.keras.optimizers.SGD(
        #     lr = learning_rate ,
        #     momentum = momentum,
        #     decay = decay_rate,
        #     nesterov = False
        # )

        optimizer = tf.keras.optimizers.Adam( lr=0.0001, decay=1e-6 )
        # optimizer = tf.keras.optimizers.Adam( clipvalue = 0.5 )

        model.compile(
            # loss    = 'categorical_crossentropy',
            loss      = 'mse',
            optimizer = optimizer,
            metrics   = [   'accuracy'   ]
        )

        return model


    def closeall( self ):
        vtrades = self.api.request(trades.TradesList(self.aid))['trades']
        for trade in vtrades:
            self.api.request(trades.TradeClose( self.aid ,
                tradeID = trade['id'],
                data = { "units" : 'ALL' }
            ))

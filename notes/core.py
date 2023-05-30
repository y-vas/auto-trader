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


from core import Core
from jict import walk
import sys, numpy as np

srv = Core()
srv.closeall()
model = srv.model(1)

for mod, valid, test, last in srv.df( split = False ):
    # y = model.predict( np.array([last[0]]) )
    y = model.predict( valid[0] )

    for x, f in zip(y,valid[1]):
        print(x[0],f[0])

    # if not TEST_PROC:
    #     if buy > ORDER_ACT and sell < 0.5:
    #         srv.order( ins , True , WASTE_FOR_POSITION )
    #     if buy < 0.5 and sell > ORDER_ACT:
    #         srv.order( ins , False , WASTE_FOR_POSITION )

    # if buy > 0.9 and sell < 0.5:
    #     if x[0] != 1: loss+=1
    #     print( 'buy', 'loss' if x[0] != 1 else 'win ' ,'|', round(i[0],4), round(i[1],4) , i )
    # elif buy < 0.5 and sell > 0.9:
    #     if x[1] != 1: loss+=1
    #     print( 'sell', 'loss' if x[1] != 1 else 'win' ,'|', round(i[0],4), round(i[1],4) , i)
    # else:
    #     skip += 1
    #     print( '---', x,  round(i[0],4), round(i[1],4) , i)

    exit()

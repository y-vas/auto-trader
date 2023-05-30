import matplotlib.pyplot as plt
import os, numpy as np
from core import Core
from jict import jict
thispath = os.path.dirname(os.path.realpath(__file__))

net = Core()

model = None
for ep in range(100):
    for mod, valid, test, last in net.df( split = False ):
        dfX  , dfy  = valid
        dftX , dfty = test

        if dfX.shape[0] == 0:
            continue

        if model == None:
            model = net.model( dfX.shape[1:] , 4 )

        model.fit(
            dfX, dfy,
            batch_size = 64, epochs = 10 ,
            validation_split = 0.03,
        )

        model.save( net.mdl+str(ep) )

    net.prod= jict( thispath + f'/source/progres-{ep}.json' )


exit()
net.train()
# net.specific_train('EUR_USD')

# y = model.predict( valid[0] )

# for x, f in zip(y,valid[1]):
#     print(x[0],f[0])

# plt.plot(df.index, df['c'], '-b', label='close')
# plt.plot(df.index, df['SAREXT'], '-g', label='sar')

# # df['bshort'] = np.where(df['bshort'] == 1,df['c'],0)
# # df['sshort'] = np.where(df['sshort'] == 1,df['c'],0)
# # plt.plot(df.index, df['bshort'], 'g2', label='buy')
# # plt.plot(df.index, df['sshort'], 'r1', label='sell')

# plt.show()

# val_loss starts increasing, val_acc starts decreasing
# (means model is cramming values not learning)

# val_loss starts increasing, val_acc also increases.
# (could be case of overfitting or diverse probability values in
# cases softmax is used in output layer)

# val_loss starts decreasing, val_acc starts increasing
# (Correct, means model build is learning and working fine)

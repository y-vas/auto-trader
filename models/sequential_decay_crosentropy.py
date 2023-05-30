import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout, LSTM, BatchNormalization


class SequentialDecayEntropy(Sequential):
    """
    Extencion of the exiting secuential model
    that adds the required layers to learn 
    and to normalize the batches
    """

    def setShape(self , shape , output = 2 ):
        self.add(LSTM( 162, input_shape= shape , return_sequences=True ))
        self.add(Dropout(0.2))
        self.add(BatchNormalization())

        self.add(LSTM( 128, return_sequences=True ))
        self.add(Dropout(0.1))
        self.add(BatchNormalization())
\
        self.add(LSTM( 64 ))
        self.add(Dropout(0.1))
        self.add(BatchNormalization())

        self.add(Dense( 32 , activation = 'relu' ))
        self.add(Dropout( 0.1 ))

        self.add(Dense( 64 , activation = 'relu' ))
        self.add(Dense( 128, activation = 'relu' ))

        self.add(Dense( units = output , activation = 'linear' ))
        optimizer = tf.keras.optimizers.Adam( lr=0.0001, decay=1e-6 )

        model.compile(
            loss    = 'categorical_crossentropy',
            optimizer = optimizer,
            metrics   = [   'accuracy'   ]
        )
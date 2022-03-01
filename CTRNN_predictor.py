import keras
import tensorflow as tf
import CTRNN_layers
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

class CTRNN:
    def __init__(
        self,
        rnn_layer,
        units_vec,
        epochs=None,
        t_vec=None,
        connectivity='dense',
        verbose=1,
        optimizer='adam',
        loss='mse',
        patience=None,
        batch_size=16,
        show_plot=True,
        lr_schedule=False,
        alpha=0.2,
        timestep=20,
        epoch_step=20,
        ):
        self.rnn_layer = getattr(CTRNN_layers, rnn_layer)
        self.units_vec = units_vec
        self.verbose = verbose
        self.optimizer = optimizer
        self.loss = loss
        self.patience = patience
        self.callback = []
        self.batch_size = batch_size
        self.epochs = epochs
        self.show_plot = show_plot
        self.t_vec = t_vec
        self.connectivity = connectivity
        self.timestep = timestep
        self.alpha = alpha
        self.epoch_step = epoch_step
        self.lr_schedule = lr_schedule
    
    def _scheduler(self, epoch, lr):
        
       if (epoch+1) % self.epoch_step == 0 and (epoch+1) <= 0.8*self.epochs:
           print(epoch+1, lr * tf.math.exp(-self.alpha))
           return lr * tf.math.exp(-self.alpha)
       else:
           return lr
    
    def _preprocessing(self, X, y=[]):
      
        X_stack = np.zeros((X.shape[0]-self.timestep+1, self.timestep, X.shape[1]))
        if y != []:
            y_stack = np.zeros((y.shape[0]-self.timestep+1, y.shape[1]))
        for i in range(X.shape[0]-self.timestep+1):
            X_stack[i, :, :] += X[i: i+self.timestep, :]
            if y != []:
                y_stack[i, :] += y[i+self.timestep-1, :]
        if y != []:
            return X_stack, y_stack
        else:
            return X_stack
    
    def _creat_model(self, input_shape, output_shape):
        
        if self.rnn_layer.__name__ == 'SimpleCTRNN':
            model = keras.Sequential(
                [keras.Input(shape=input_shape)] + \
                [self.rnn_layer(units=units, return_sequences=True)
                                         for units in self.units_vec[:-1]] + \
                [
                self.rnn_layer(units=self.units_vec[-1], return_sequences=False),
                layers.Dense(output_shape)]
                )
        else:
            model = keras.Sequential(
                [keras.Input(shape=input_shape),
                self.rnn_layer(units_vec=self.units_vec, t_vec=self.t_vec,
                               connectivity=self.connectivity),
                layers.Dense(output_shape)]
                )
        
        if self.verbose > 1:
            model.summary()
        
        return model

    def fit(self, X, y, sample_weight=None, return_shape=False):
        
        X, y = self._preprocessing(X, y)
        
        input_shape = X.shape[1:]
        output_shape = y.shape[1]
        
        self.model = self._creat_model(input_shape, output_shape)
        self.model.compile(self.optimizer, self.loss)
        if self.patience != None:
            self.callback.append(keras.callbacks.EarlyStopping(monitor='loss',
                      patience=self.patience, verbose=self.verbose, mode='min'))
        
        if self.lr_schedule:
            self.callback.append(keras.callbacks.LearningRateScheduler(
                                                               self._scheduler))
        history = self.model.fit(X, y, batch_size=self.batch_size, 
                            callbacks=self.callback, epochs=self.epochs,
                            verbose=self.verbose+1, shuffle=False)
        
        if self.show_plot:
            plt.figure()
            plt.plot(history.history['loss'], label='loss')
            plt.legend()
            plt.show()
        
        if return_shape:
            return input_shape, output_shape
        
    def predict(self, X):
        
        X = self._preprocessing(X)
        y_pred = self.model.predict(X)
        
        return y_pred
    
    def save(self, save_path):
        
        self.model.save_weights(save_path)
    
    def load(self, save_path, input_shape, output_shape):
        
        self.model = self._creat_model(input_shape, output_shape)
        self.model.load_weights(save_path)
        
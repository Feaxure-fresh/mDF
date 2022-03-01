import keras
import matplotlib.pyplot as plt
from deepforest import CascadeForestRegressor
from keras.models import Sequential
from keras.layers import Dense
from CTRNN_predictor import CTRNN

class Regressor:
    def __init__(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        y_scaler,
        timestep=20,
        ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.y_scaler = y_scaler
        self.timestep = timestep - 1
    
    def _predict_all(self, model, data_version):
        y_pred = self.y_scaler.inverse_transform(model.predict(self.x_test))
        y_test = self.y_scaler.inverse_transform(self.y_test)
        train_pred = self.y_scaler.inverse_transform(model.predict(self.x_train))
        y_train = self.y_scaler.inverse_transform(self.y_train)
        
        if data_version == 1:
            return y_pred[self.timestep:, :], y_test[self.timestep:, :], \
                   train_pred[self.timestep:, :], y_train[self.timestep:, :]
        else:
            return y_pred, y_test[self.timestep:, :], \
                   train_pred, y_train[self.timestep:, :]
        
    def DF(self, save_path=None):
        model = CascadeForestRegressor(use_predictor=True, n_estimators=1,
                                       n_bins=255, n_trees=500, criterion='mse',
                                       n_tolerant_rounds=1)
        model.fit(self.x_train, self.y_train)
        
        if save_path != None:
            model.save(save_path)

        return self._predict_all(model, data_version=1)

    
    def GACTRNN(self, save_path=None):
        model = CTRNN(rnn_layer='GACTRNN', units_vec=[64, 32, 16, 8],
                      t_vec=[1, 16, 64, 216], epochs=300, patience=20,
                      verbose=1, lr_schedule=True, timestep=self.timestep+1)
        model.fit(self.x_train, self.y_train)
        
        if save_path != None:
            model.save(save_path)

        return self._predict_all(model, data_version=2)
    
    def mDF(self, save_path=None):
        model = CascadeForestRegressor(use_predictor=True, n_estimators=1,
                                       n_bins=255, n_trees=500, criterion='mse',
                                       n_tolerant_rounds=1, predictor_no_binner=True)
        predictor = CTRNN(rnn_layer='GACTRNN', units_vec=[64, 32, 16, 8],
                          t_vec=[1, 16, 64, 216], epochs=300, patience=20,
                          verbose=1, lr_schedule=True, timestep=self.timestep+1)
        model.set_predictor(predictor)
        model.fit(self.x_train, self.y_train)
        
        if save_path != None:
            model.save(save_path)

        return self._predict_all(model, data_version=2)
        
    def BPNN(self, save_path=None, show_plot=False):
        model = Sequential()
        model.add(Dense(units=512, input_dim=8, activation='tanh'))
        model.add(Dense(units=256, activation='tanh')) 
        model.add(Dense(units=128, activation='tanh'))
        model.add(Dense(units=64, activation='tanh'))
        model.add(Dense(units=32, activation='tanh'))
        model.add(Dense(units=16, activation='tanh'))
        model.add(Dense(units=8, activation='tanh'))
        model.add(Dense(units=4, activation='tanh'))
        model.add(Dense(units=2))
        model.summary()
        
        model.compile(optimizer='adam', loss='mse')
        callback = keras.callbacks.EarlyStopping(monitor='loss', patience=10, 
                                            verbose=2, mode='min')
        history = model.fit(self.x_train, self.y_train, epochs=200, batch_size=32,
                            validation_data=(self.x_test, self.y_test),
                            callbacks=[callback], verbose=2, shuffle=False)
        
        if show_plot:
            fig1 = plt.figure(figsize=(12, 9))
            ax1=fig1.add_subplot(111)
            ax1.plot(history.history['loss'],linestyle="-")
            ax1.plot(history.history['val_loss'],linestyle="--")
            ax1.set_xlabel('训练次数')
            ax1.set_ylabel('损失值')
            plt.legend(["Train mse","Test mse"])
    
        if save_path != None:
            model.save(save_path)

        return self._predict_all(model, data_version=1)
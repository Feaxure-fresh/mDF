import os
import math
import numpy as np
import pandas as pd
from matplotlib.pylab import mpl
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from deepforest import CascadeForestRegressor 
from models import Regressor
from CTRNN_predictor import CTRNN


mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.size'] = 30
color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


class train_utils(object):
    def __init__(self, args):
        self.emg_path = args.emg_path
        self.force_path = args.force_path
        self.test_size = args.test_size
        self.model = args.model
        self.save_dir = args.save_dir
        self.timestep = args.timestep
        
    @staticmethod
    def _print_metrics(y_pred, y_test, model_name=''):
        '''
        Print metrics (mse and Pearson correlation coefficient) for result.
        '''
        FDS_mse = mean_squared_error(y_test[:,0], y_pred[:,0])
        EDI_mse = mean_squared_error(y_test[:,1], y_pred[:,1])
        print('\n%s FDS的mse: %.06f' % (model_name, FDS_mse))
        print('%s EDI的mse: %.06f' % (model_name, EDI_mse))
        
        y_test = pd.DataFrame(y_test)
        y_pred = pd.DataFrame(y_pred)
        df = pd.concat([y_test, y_pred], axis=1, ignore_index=True)
        df.columns = ['FDS', 'EDI', 'pred_FDS', 'pred_EDI']
        corr = df.corr(method = 'pearson')
        print('\nFDS pearson corr: %.04f' % corr.loc['FDS', 'pred_FDS'])
        print('EDI pearson corr: %.04f' % corr.loc['EDI', 'pred_EDI'])
    
    def load_result(self, model_path=[], model_index=[]):
        '''
        Load existing trained model and show results.
            
            model_path: list, paths for trained models.
            model_index: list, choices in 'DF', 'mDF' and 'GACTRNN'. Names for items in model_path.
        '''
        if len(model_path) == 0:
            if os.path.exists(os.path.join(self.save_dir, 'DF')):
                model_path.append(os.path.join(self.save_dir, 'DF'))
                model_index.append('DF')
            if os.path.exists(os.path.join(self.save_dir, 'GACTRNN.h5')):
                model_path.append(os.path.join(self.save_dir, 'GACTRNN.h5'))
                model_index.append('GACTRNN')
            if os.path.exists(os.path.join(self.save_dir, 'mDF')):
                model_path.append(os.path.join(self.save_dir, 'mDF'))
                model_index.append('mDF')
            
            if len(model_path) == 0:
                raise Exception("No model is saved in saving directory.")
            
        if len(model_path) != len(model_index):
            raise Exception("The length for model_path and model_index must be same.")
        
        emg = pd.read_excel(self.emg_path, header=None)
        force = pd.read_excel(self.force_path, header=None)
        
        x_scaler = preprocessing.MinMaxScaler()
        y_scaler = preprocessing.MinMaxScaler()
        x = x_scaler.fit_transform(emg)
        y = force.values
        y_scaler.fit(force)
        
        test_size = math.floor(self.test_size * x.shape[0])
        start_num = math.floor((1 - self.test_size) * x.shape[0])
        x_test = x[start_num:start_num+test_size, :]
        y_test = y[start_num:start_num+test_size, :]
            
        timestep = self.timestep - 1
        fig1 = plt.figure(num=1, figsize=(12, 9))
        ax1 = fig1.add_subplot(111)
        
        fig2 = plt.figure(num=2, figsize=(12, 9))
        ax2 = fig2.add_subplot(111)
        
        y_test = y_test[timestep:, :]
        ax1.plot(y_test[:, 0], linestyle="--", color=color[1], linewidth=3, label='真实值')
        ax2.plot(y_test[:, 1], linestyle="--", color=color[1], linewidth=3, label='真实值')
        
        for i, index in enumerate(model_index):
            if index == 'DF':
                model = CascadeForestRegressor()
                model.load(model_path[i])
                
                y_pred = model.predict(x_test)
                y_pred = y_scaler.inverse_transform(y_pred[timestep:, :])
                
                ax1.plot(y_pred[:,0], linestyle="-", color=color[0], label=index)
                ax2.plot(y_pred[:,1], linestyle="-", color=color[0], label=index)
                
                self._print_metrics(y_pred, y_test, model_name=index)
            elif index == 'GACTRNN':
                model = CTRNN(rnn_layer='GACTRNN', units_vec=[64, 32, 16, 8],
                              t_vec=[1, 16, 64, 216], timestep=timestep+1)
                model.load(model_path[i], (timestep+1, 8), 2)
                
                y_pred = model.predict(x_test)
                y_pred = y_scaler.inverse_transform(y_pred)
                
                ax1.plot(y_pred[:,0], linestyle="-", color=color[2], label=index)
                ax2.plot(y_pred[:,1], linestyle="-", color=color[2], label=index)
                
                self._print_metrics(y_pred, y_test, model_name=index)
            elif index == 'mDF':
                model = CascadeForestRegressor()
                predictor = CTRNN(rnn_layer='GACTRNN', units_vec=[64, 32, 16, 8],
                              t_vec=[1, 16, 64, 216], timestep=timestep+1)
                model.load(model_path[i], predictor)
                
                y_pred = model.predict(x_test)
                y_pred = y_scaler.inverse_transform(y_pred)
                
                ax1.plot(y_pred[:,0], linestyle="-", color=color[3], label=index)
                ax2.plot(y_pred[:,1], linestyle="-", color=color[3], label=index)
                
                self._print_metrics(y_pred, y_test, model_name=index)
            else:
                raise Exception("model_index: list, choices in 'DF', 'mDF' and 'GACTRNN'.")
        
        ax1.set_xlabel('帧/个')
        ax1.set_ylabel('FDS/N')
        ax2.set_xlabel('帧/个')
        ax2.set_ylabel('EDI/N')
        ax1.legend(fontsize = 15)
        ax2.legend(fontsize = 15)
        plt.show()
            
    def train(self):
        '''
        Training Process.
        '''
        emg = pd.read_excel(self.emg_path, header=None)
        force = pd.read_excel(self.force_path, header=None)
        
        x_scaler = preprocessing.MinMaxScaler()
        y_scaler = preprocessing.MinMaxScaler()
        x = x_scaler.fit_transform(emg)
        y = y_scaler.fit_transform(force)
        
        test_size = math.floor(self.test_size * x.shape[0])
        start_num = math.floor((1 - self.test_size) * x.shape[0])
        x_test = x[start_num:start_num+test_size, :]
        y_test = y[start_num:start_num+test_size, :]
        x_train = np.delete(x, range(start_num, start_num+test_size), 0)
        y_train = np.delete(y, range(start_num, start_num+test_size), 0)
    
        regressor = Regressor(x_train, y_train, x_test,
                              y_test, y_scaler, timestep=self.timestep)
        model = getattr(regressor, self.model)
        
        if self.save_dir == None:
            save_path = None
        else:
            if self.model == 'DF':
                save_path = os.path.join(self.save_dir, 'DF')
            elif self.model == 'mDF':
                save_path = os.path.join(self.save_dir, 'mDF')
            elif self.model == 'GACTRNN':
                save_path = os.path.join(self.save_dir, 'GACTRNN.h5')
        y_pred, y_test, train_pred, y_train = model(save_path=save_path)
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111)
        ax.plot(y_pred[:,0], linestyle="-", color=color[0], label='预测值')
        ax.plot(y_test[:,0], linestyle="--", color=color[1], label='真实值')
        ax.set_xlabel('帧/个')
        ax.set_ylabel('FDS/N')
        plt.legend(fontsize = 15)
    
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111)
        ax.plot(y_pred[:,1],linestyle="-", color=color[0], label='预测值')
        ax.plot(y_test[:,1],linestyle="--", color=color[1], label='真实值')
        ax.set_xlabel('帧/个')
        ax.set_ylabel('EDI/N')
        plt.legend(fontsize = 15)
        plt.show()
        
        self._print_metrics(y_pred, y_test)

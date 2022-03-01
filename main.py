import os
import argparse
from train_utils import train_utils

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # basic parameters
    parser.add_argument('--model', type=str, choices=['DF', 'mDF', 'GACTRNN'], default='mDF', help='the name of the model')
    parser.add_argument('--emg_path', type=str, default=".\test_data\subjectB_emg.xlsx", help='the path of EMG data')
    parser.add_argument('--force_path', type=str, default=".\test_data\subjectB_force.xlsx", help='the path of force data')
    parser.add_argument('--test_size', type=float, default=0.2, help='the name of the data')
    parser.add_argument('--save_dir', type=str, default=".\saving", help='the directory for saving model')
    parser.add_argument('--timestep', type=int, default=20, help='timestep for mDF and GACTRNN')
    parser.add_argument('--load', type=bool, default=False, help='switch mode for training and loading models')
    args = parser.parse_args()
    return args
    

if __name__ == '__main__':
    args = parse_args()
    if args.save_dir != None and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    trainer = train_utils(args)
    if not args.load:
        trainer.train()
    else:
        trainer.load_result()

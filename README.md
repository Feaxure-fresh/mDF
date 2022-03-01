# mDF
Improved Deep Forest (mDF) model for hand muscle force assessment based on multi-channel EMG signals.

Modified based on [Deep-Forest](https://github.com/LAMDA-NJU/Deep-Forest) and [GACTRNN](https://github.com/heinrichst/GACTRNN).

# Installation
### Prerequisites
*  python3 (>=3.7) with the development headers.
*  TensorFlow (=2.4.1)

### Train with mDF
```
python main.py
```
### Train with Deep Forest
```
python main.py --model DF
```
### Train with GACTRNN
```
python main.py --model GACTRNN
```
### Load trained models
```
python main.py --load True --save_dir ./trained_models
```

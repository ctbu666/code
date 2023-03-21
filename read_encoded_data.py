
from data_provider import *


path_in = './dataset/encoded/data_encoded.pkl'
with open(path_in, 'rb') as f:
    images_train, images_test = pickle.load(f)
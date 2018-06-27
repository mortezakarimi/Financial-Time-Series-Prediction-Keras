import matplotlib.pyplot as plt
import numpy as np
from utils import wavelet_denoising
import pandas as pd

dataset = pd.read_csv('000001.SS.csv', usecols=[ 4])
Close_list = dataset[['Close']].mean(axis=1)
rev1 = wavelet_denoising(Close_list)
# namayeshe close price avaliye
plt.figure(1)
plt.plot(Close_list, label='Closing Stocks data')
plt.xlim(xmin=0, xmax=len(Close_list))
plt.ylim(ymin=1500, ymax=5500)
plt.xlabel('Days')
plt.ylabel('value of stocks')
plt.legend()
plt.savefig('Closing price of SHI.png')
plt.show()

# namayesh close price bad az wavelet reconstruction
plt.figure(2)
plt.plot(rev1, label='Closing Stocks data after wavelet reconstruction')
plt.xlim(xmin=0, xmax=len(rev1))
plt.ylim(ymin=1500, ymax=5500)
plt.xlabel('Days')
plt.ylabel('value of stocks')
plt.legend()
plt.savefig('Closing price of SHI after wavelet reconstruction.png')
plt.show()

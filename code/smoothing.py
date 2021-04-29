""" 
This file is used to generate the plot used in the report

"""
import pandas as pd
from scipy.ndimage.filters import uniform_filter1d
import matplotlib.pyplot as plt


data_col = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv('data/track1data/driving_log.csv', names= data_col)
    # smoothing the steering angle to avoid jettering in the output
a = data['steering'][0:1000]
b = uniform_filter1d(a, size=25)

plt.plot(a, color = 'r' , linestyle = 'dotted', label = "recorded")
plt.plot(b, color = 'b',  label = "smoothed")

plt.title("Smoothing of Steering Angle")
plt.xlabel("Index")
plt.ylabel("Steering Angle")
plt.legend(loc="upper left")

plt.show()



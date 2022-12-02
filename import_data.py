import pandas as pd
from datetime import datetime
import numpy as np
def split_trajs(npdata, sl, win_len):

    #### Build the outputs/labels
    # Sigmoid boolean
    percent_close = .001 #difference of (+- percent_close * close) will map to +- .88 in sigmoid
    output = np.array([npdata[i]-npdata[i-1] for i in range(win_len+1, len(npdata))])[:,1]
    scale = 2 / (percent_close * np.array([npdata[i-1] for i in range(win_len+1, len(npdata))])[:,1])
    output = 1 / (1 + np.exp(-1*output*scale)) #sigmoid function. Scaled to put 88% at threshold eps
    #output = [ for i in len(output_price)]

    # Close differential
    # output = np.array([npdata[i]-npdata[i-1] for i in range(win_len+1, len(npdata))])[:,1]
    
    # Close price
    # output = np.array([npdata[i] for i in range(win_len, len(npdata))])[:,1]

    ### Build the inputs/features
    #Differentials TODO: don't take differential of volume
    inputs_window = np.array([npdata[i-win_len:i]-npdata[i-1-win_len:i-1] for i in range(win_len+1,len(npdata))])
    inputs_window = inputs_window.reshape(inputs_window.shape[0], -1)
    if len(inputs_window) == 384:
        return inputs_window.reshape(4,sl,5 * win_len), output.reshape(4,sl)
    else:
        npdata_filt = inputs_window[:sl*3]
        output_filt = output[:sl*3]
        return npdata_filt.reshape(3,sl,5 * win_len), output_filt.reshape(3,sl)
    
    #Prices
    '''inputs_window = np.array([npdata[i-win_len:i] for i in range(win_len,len(npdata))])
    inputs_window = inputs_window.reshape(inputs_window.shape[0], -1)
    if len(inputs_window) == 384:
        return inputs_window.reshape(4,sl,5 * win_len), output.reshape(4,sl)
    else:
        npdata_filt = inputs_window[:sl*3]
        output_filt = output[:sl*3]
        return npdata_filt.reshape(3,sl,5 * win_len), output_filt.reshape(3,sl)'''


def grab_amazon_data(filename = "AMZN_2022.csv", sl = 96, win_len = 6):
    amzn_data = pd.read_csv(filename).set_index("DateTime").drop('Unnamed: 0', axis = 1, inplace = False).drop_duplicates()
    date_index = [datetime.strptime(i, "%Y-%m-%d %H:%M:%S") for i in list(amzn_data.index)]
    amzn_data.index = date_index

    unique_dates = pd.unique([i.date() for i in date_index])
    date_locs = np.array([[i.date() == ud for i in date_index] for ud in unique_dates])
    
    trajs = [split_trajs(amzn_data.loc[date_locs[udidx]].to_numpy(), sl, win_len) for udidx in range(len(unique_dates))]

    input_windowed = np.array([i for j in [traji for traji, output in trajs] for i in j])
    output_windowed = np.array([i for j in [output for traji, output in trajs] for i in j])
    
    print(1)

    return input_windowed, output_windowed


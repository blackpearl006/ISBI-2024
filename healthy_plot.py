#importing necessary libaraies
import os
import gc
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
        
def state_rec_plot(df, name=None, tau=1, dim = 2):
    '''This function implemenst the idea of recurrence plot the pdist function takes the L2 norm (Euclidean distance)
    between the states and the sqaureform function craetes a symetric data matrix from the pairwise distances.
    Essentailly plotting the states of the time series in a different dimension.
    Input : 1 Dimensional column vector 
    Output : PNG image
    the patterns and inferences from the image is highly dependent of tau and dims values (important)
    dim : optimal embedding dimension (cao'a algo , false neighbourhood algorithm)
    tau : time lag (first minima of autocorrelation)
    '''
    df = df.reshape(-1,1)
    tuple_vector = [[ df[i+j*tau][0] for j in range(dim)] for i in range(len(df) - dim*tau)]
    states = pdist(tuple_vector)
    m_states =  squareform(states)  

    plt.figure(figsize=(20, 20))
    plt.imshow(m_states, cmap='gray')
    plt.savefig(name+'state.png') 
    plt.close()
    gc.collect()
    return ''

def get_data(network):
    '''
    function to get the data from the csv files
    data is returned as a dictionary object with the keys being the 2 claases 0 (healthy) and 1 (mci) which in turn is a dictionary
    the 2 classes have all subjects as thier key and values are a numpy array of the shape (number of rois, 187)
    Input : folder path to the network folder which has 2 folders healthy and mci which in turn has xx subjects .csv folders
    Output : dictionary[ class { 0, 1} ][ subject_name ][ ROI number ]
    '''
    data_dir = f'/Users/ninad/Documents/_CBR/Data/ROI CSV files/{network}_ts/healthy'
    data = {}
    for j in os.listdir(data_dir):
        data[j] = np.array(pd.read_csv(os.path.join(data_dir,j), header=None).values)
    return data

network = 'default_mode'
# df = pd.read_csv('/Users/ninad/Documents/_CBR/Scripts/Recurrence plots/first10_healthy.csv')
# df = pd.read_csv('/Users/ninad/Documents/_CBR/Scripts/Recurrence plots/second10_healthy.csv')
# df = pd.read_csv('/Users/ninad/Documents/_CBR/Scripts/Recurrence plots/third10_healthy.csv')
df = pd.read_csv('/Users/ninad/Documents/_CBR/Scripts/Recurrence plots/last4_healthy.csv')

data = get_data(network=network)
roi_req = [i for i in range(31,35)]
roi_index = [i-1 for i in roi_req]

class_dir = '/Users/ninad/Documents/_CBR/Data/ROIwisehealthy'

os.makedirs(class_dir, exist_ok=True)
for roi in roi_index:
    roi_dir = f'{class_dir}/Roi{roi+1}'
    os.makedirs(roi_dir, exist_ok=True)

excluded_list = []

for roi in roi_index:
    for sub_name in data.keys():
        subject_data = df[(df['subject'] == sub_name) & (df['ROI'] == roi+1)]
        try:
            dim = int(subject_data['DIM'].iloc[0])
            tau = int(subject_data['Tau'].iloc[0])
            time_data = data[sub_name][roi]
            if tau * dim < 170:
                plt.figure(figsize=(20, 20))
                state_rec_plot(time_data, name=f'{class_dir}/Roi{roi+1}/sub_{sub_name[8:-4]}_roi{roi+1}_dim{dim}_tau{tau}', tau=tau, dim=dim)
                # state_rec_plot(time_data, name=f'{class_dir}/Roi{roi+1}/sub_{sub_name[8:-4]}_roi{roi+1}', tau=tau, dim=dim)
                plt.close()
            else:
                excluded_list.append((sub_name, roi+1, dim, tau))
        except Exception as e:
            excluded_list.append((sub_name, roi+1, 'out of bound'))

if len(excluded_list)>1:
    file_path = 'healthy_excluded.txt'
    with open(file_path, 'w') as file:
        file.write('\n'.join(map(str, excluded_list)))


#importing necessary libaraies
import os
import gc
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
    To obtain a 224 x 224 image with figsize=(4,4) we need dpi = 224/4 = 56
    '''
    df = df.reshape(-1,1)
    tuple_vector = [[ df[i+j*tau][0] for j in range(dim)] for i in range(len(df) - dim*tau)]
    states = pdist(tuple_vector)
    m_states =  squareform(states)  

    fig = plt.figure(frameon=False)
    fig.set_size_inches(4,4)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(m_states, aspect='auto', cmap='gray')
    plt.savefig(name+'state.png',dpi=56, bbox_inches='tight')
    plt.close()
    gc.collect()
    return ''


def get_data(network, subject_class):
    '''
    function to get the data from the csv files
    data is returned as a dictionary object with the keys being the 2 claases 0 (healthy) and 1 (mci) which in turn is a dictionary
    the 2 classes have all subjects as thier key and values are a numpy array of the shape (number of rois, 187)
    Input : folder path to the network folder which has 2 folders healthy and mci which in turn has xx subjects .csv folders
    Output : dictionary[ class { 0, 1} ][ subject_name ][ ROI number ]
    '''
    data_dir = f'/Users/ninad/Documents/_CBR/Data/ROI CSV files/{network}_ts/{subject_class}'
    data = {}
    for j in os.listdir(data_dir):
        data[j] = np.array(pd.read_csv(os.path.join(data_dir,j), header=None).values)
    return data

group = 1
subject_class = 'mci'
networks = ['default_mode','cerebellum','frontoparietal','occipital','cingulo-opercular', 'sensorimotor']
network_CSV_name = [ 'DMN','CB','FP','OP','CO','SM']
for network in range(len(networks)):
    df = pd.read_csv(f'/Users/ninad/Documents/_CBR/Scripts/Recurrence plots/CSV files/{network_CSV_name[network]}{group}_{subject_class}.csv')

    data = get_data(network=networks[network],subject_class=subject_class)
    roi_req = [i for i in range(1, 11)]
    roi_index = [i-1 for i in roi_req]

    class_dir = f'path/to/{networks[network]}/{subject_class}'

    #creating output files

    os.makedirs(class_dir, exist_ok=True)
    # uncomment to have recurrence plots for each ROI in different folders
    # for roi in roi_index:
    #     roi_dir = f'{class_dir}/Roi{roi+1}'
    #     os.makedirs(roi_dir, exist_ok=True)

    excluded_list = []

    for roi in roi_index:
        for sub_name in data.keys():
            subject_data = df[(df['subject'] == sub_name) & (df['ROI'] == roi+1)]
            try:
                dim = int(subject_data['DIM'].iloc[0])
                tau = int(subject_data['Tau'].iloc[0])
                time_data = data[sub_name][roi]
                if tau * dim < 170:
                    # state_rec_plot(time_data, name=f'{class_dir}/Roi{roi+1}/sub_{sub_name[8:-4]}_roi{roi+1}_dim{dim}_tau{tau}', tau=tau, dim=dim)
                    state_rec_plot(time_data, name=f'{class_dir}/sub_{sub_name[-6:-4]}_roi{roi+1}', tau=tau, dim=dim)
                else:
                    excluded_list.append((sub_name, roi+1, dim, tau))
            except Exception as e:
                excluded_list.append((sub_name, roi+1, 'out of bound'))

    if len(excluded_list)>1:
        file_path = f'mci_excluded{networks[network]}_{network+1}.txt'
        with open(file_path, 'w') as file:
            file.write('\n'.join(map(str, excluded_list)))

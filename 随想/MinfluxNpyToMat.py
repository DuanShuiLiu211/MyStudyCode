import os
import numpy as np
from scipy import io


def read_npy(folder_name):
    data_list = []
    file_list = os.listdir(folder_name)
    num = []
    for n, file in enumerate(file_list):
        if file.split('.')[-1] == 'npy':
            data_list.append(np.load(os.path.join(folder_name, file)))
            num.append(n)
    return data_list, file_list, num


def get_itr_4(data):
    loc_list = []
    efo_list = []
    dcr_list = []
    for itr in data:
        if itr['vld']:
            itr_list = itr[0]
            loc = itr_list[4]['loc']
            efo = itr_list[4]['efo']
            dcr = itr_list[4]['dcr']
            if not (np.isnan(loc[0]) or np.isnan(loc[1])):
                loc_list.append(loc)
            else:
                loc_list.append(None)
            if not np.isnan(efo):
                efo_list.append(efo)
            else:
                efo_list.append(None)
            if not np.isnan(dcr):
                dcr_list.append(dcr)
            else:
                dcr_list.append(None)
    return loc_list, efo_list, dcr_list


def get_itr_9(data):
    loc_list = []
    efo_list = []
    dcr_list = []
    for itr in data:
        if itr['vld']:
            itr_list = itr[0]
            loc = itr_list[9]['loc']
            efo = itr_list[9]['efo']
            dcr = itr_list[9]['dcr']
            if not (np.isnan(loc[0]) or np.isnan(loc[1]) or np.isnan(loc[2])):
                loc_list.append(loc)
            else:
                loc_list.append(None)
            if not np.isnan(efo):
                efo_list.append(efo)
            else:
                efo_list.append(None)
            if not np.isnan(dcr):
                dcr_list.append(dcr)
            else:
                dcr_list.append(None)
    return loc_list, efo_list, dcr_list


def save_mat(loc_list, efo_list, dcr_list, save_path):
    io.savemat(save_path, {'loc': loc_list, 'efo': efo_list, 'dcr': dcr_list})


if __name__ == '__main__':
    folder_name1 = r'W:\桌面\1'
    path1 = r'W:\桌面'
    data_list1, file_list1, num1 = read_npy(folder_name1)
    for i, j in enumerate(num1):
        data1 = data_list1[i]
        loc_list1, efo_list1, dcr_list1 = get_itr_9(data1)
        save_path1 = path1 + r'\{}.mat'.format(file_list1[j])
        save_mat(loc_list1, efo_list1, dcr_list1, save_path1)

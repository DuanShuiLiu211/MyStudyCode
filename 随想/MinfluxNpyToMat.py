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
    tic_list = []
    loc_list = []
    efo_list = []
    cfr_list = []
    dcr_list = []
    for itr in data:
        if itr['vld']:
            itr_list = itr[0]
            tic = itr_list[4]['tic']
            loc = itr_list[4]['loc']
            efo = itr_list[4]['efo']
            cfr = itr_list[4]['cfr']
            dcr = itr_list[4]['dcr']
            if not np.isnan(tic):
                tic_list.append(tic)
            else:
                tic_list.append(None)
            if not (np.isnan(loc[0]) or np.isnan(loc[1])):
                loc_list.append(loc)
            else:
                loc_list.append(None)
            if not np.isnan(efo):
                efo_list.append(efo)
            else:
                efo_list.append(None)
            if not np.isnan(cfr):
                cfr_list.append(cfr)
            else:
                cfr_list.append(None)
            if not np.isnan(dcr):
                dcr_list.append(dcr)
            else:
                dcr_list.append(None)
    return tic_list, loc_list, efo_list, cfr_list, dcr_list


def get_itr_9(data):
    tic_list = []
    loc_list = []
    efo_list = []
    cfr_list = []
    dcr_list = []
    for itr in data:
        if itr['vld']:
            # dtype: [('itr', '<i4'), ('tic', '<u4'), ('loc', '<f8', (3,)), ('eco', '<i4'), ('ecc', '<i4'),
            # ('efo', '<f8'), ('efc', '<f8'), ('sta', '<i4'), ('cfr', '<f8'), ('dcr', '<f8'), ('ext', '<f8', (3,)),
            # ('gvy', '<f8'), ('gvx', '<f8'), ('eoy', '<f8'), ('eox', '<f8'), ('dmz', '<f8'), ('lcy', '<f8'),
            # ('lcx', '<f8'), ('lcz', '<f8'), ('fbg', '<f8')]
            itr_list = itr[0]
            tic = itr_list[9]['tic']
            loc = itr_list[9]['loc']
            efo = itr_list[9]['efo']
            cfr = itr_list[9]['cfr']
            dcr = itr_list[9]['dcr']
            if not np.isnan(tic):
                tic_list.append(tic)
            else:
                tic_list.append(None)
            if not (np.isnan(loc[0]) or np.isnan(loc[1]) or np.isnan(loc[2])):
                loc_list.append(loc)
            else:
                loc_list.append(None)
            if not np.isnan(efo):
                efo_list.append(efo)
            else:
                efo_list.append(None)
            if not np.isnan(cfr):
                cfr_list.append(cfr)
            else:
                cfr_list.append(None)
            if not np.isnan(dcr):
                dcr_list.append(dcr)
            else:
                dcr_list.append(None)
    return tic_list, loc_list, efo_list, cfr_list, dcr_list


def save_mat(tic_list, loc_list, efo_list, cfr_list, dcr_list, save_path):
    io.savemat(save_path, {'tic': tic_list, 'loc': loc_list, 'efo': efo_list, 'cfr': cfr_list, 'dcr': dcr_list})


if __name__ == '__main__':
    mode = 0
    if mode == 0:
        folder_name1 = r'/Volumes/GAIN-WH/Minflux/0625'
        path1 = r'/Volumes/GAIN-WH/Minflux'
        data_list1, file_list1, num1 = read_npy(folder_name1)
        for i, j in enumerate(num1):
            data1 = data_list1[i]
            tic_list1, loc_list1, efo_list1, cfr_list1, dcr_list1 = get_itr_9(data1)
            save_path1 = path1 + r'\{}.mat'.format(file_list1[j])
            save_mat(tic_list1, loc_list1, efo_list1, cfr_list1, dcr_list1, save_path1)

    elif mode == 1:
        # Import the saved data.
        # during the final iteration (denoted by the index -1) of valid events.
        mfx = np.load('\桌面\PSD95 647 RIM 594 M2.npy')
        tic1 = mfx[mfx['vld']]['itr']['tic'][:, -1]
        loc1 = mfx[mfx['vld']]['itr']['loc'][:, -1, :]
        efo1 = mfx[mfx['vld']]['itr']['efo'][:, -1]
        cfr1 = mfx[mfx['vld']]['itr']['cfr'][:, -1]
        dcr1 = mfx[mfx['vld']]['itr']['dcr'][:, -1]

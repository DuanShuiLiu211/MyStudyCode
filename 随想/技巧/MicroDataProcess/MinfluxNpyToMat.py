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


def get_data(data, dim):
    tim_list = []
    tid_list = []
    act_list = []
    itr_list = []
    tic_list = []
    loc_list = []
    eco_list = []
    ecc_list = []
    efo_list = []
    efc_list = []
    sta_list = []
    cfr_list = []
    dcr_list = []
    ext_list = []
    gvy_list = []
    gvx_list = []
    eoy_list = []
    eox_list = []
    dmz_list = []
    lcy_list = []
    lcx_list = []
    lcz_list = []
    fbg_list = []
    for idx, itr in enumerate(data):
        if itr['vld']:
            # dtype: [('itr', '<i4'), ('tic', '<u4'), ('loc', '<f8', (3,)), ('eco', '<i4'), ('ecc', '<i4'),
            # ('efo', '<f8'), ('efc', '<f8'), ('sta', '<i4'), ('cfr', '<f8'), ('dcr', '<f8'), ('ext', '<f8', (3,)),
            # ('gvy', '<f8'), ('gvx', '<f8'), ('eoy', '<f8'), ('eox', '<f8'), ('dmz', '<f8'), ('lcy', '<f8'),
            # ('lcx', '<f8'), ('lcz', '<f8'), ('fbg', '<f8')]
            tim = itr['tim']
            tid = itr['tid']
            act = itr['act']

            list = itr[0]
            itr = list[-1]['itr']
            tic = list[-1]['tic']
            loc = list[-1]['loc']
            eco = list[-1]['eco']
            ecc = list[-1]['ecc']
            efo = list[-1]['efo']
            efc = list[-1]['efc']
            sta = list[-1]['sta']
            cfr = list[-1]['cfr']
            dcr = list[-1]['dcr']
            ext = list[-1]['ext']
            gvy = list[-1]['gvy']
            gvx = list[-1]['gvx']
            eoy = list[-1]['eoy']
            eox = list[-1]['eox']
            dmz = list[-1]['dmz']
            lcy = list[-1]['lcy']
            lcx = list[-1]['lcx']
            lcz = list[-1]['lcz']
            fbg = list[-1]['fbg']

            tim_list.append([idx, tim])
            tid_list.append([idx, tid])
            act_list.append([idx, act])

            itr_list.append(itr)
            tic_list.append(tic)

            if dim == 3:
                if not (np.isnan(loc[0]) or np.isnan(loc[1]) or np.isnan(loc[2])):
                    loc_list.append(loc)
                else:
                    loc_list.append(0)
            elif dim == 2:
                if not (np.isnan(loc[0]) or np.isnan(loc[1])):
                    loc_list.append(loc)
                else:
                    loc_list.append(0)

            eco_list.append(eco)
            ecc_list.append(ecc)

            if not np.isnan(efo):
                efo_list.append(efo)
            else:
                efo_list.append(0)

            if not np.isnan(efc):
                efc_list.append(efc)
            else:
                efc_list.append(0)

            sta_list.append(sta)
            cfr_list.append(cfr)

            if not np.isnan(dcr):
                dcr_list.append(dcr)
            else:
                dcr_list.append(0)

            ext_list.append(ext)
            gvy_list.append(gvy)
            gvx_list.append(gvx)
            eoy_list.append(eoy)
            eox_list.append(eox)
            dmz_list.append(dmz)
            lcy_list.append(lcy)
            lcx_list.append(lcx)
            lcz_list.append(lcz)
            fbg_list.append(fbg)

    return tic_list, tic_list, loc_list, eco_list, ecc_list, efo_list, efc_list, sta_list, cfr_list, dcr_list, \
           ext_list, gvy_list, gvx_list, eoy_list, eox_list, dmz_list, lcy_list, lcx_list, lcz_list, fbg_list, \
           tim_list, tid_list, act_list


def save_mat(save_path, itr_list, tic_list, loc_list, eco_list, ecc_list, efo_list, efc_list, sta_list,
             cfr_list, dcr_list, ext_list, gvy_list, gvx_list, eoy_list, eox_list, dmz_list, lcy_list,
             lcx_list, lcz_list, fbg_list, tim_list, tid_list, act_list):
    io.savemat(save_path, {'itr': itr_list, 'tic': tic_list, 'loc': loc_list, 'eco': eco_list, 'ecc': ecc_list,
                           'efo': efo_list, 'efc': efc_list, 'sta': sta_list, 'cfr': cfr_list, 'dcr': dcr_list,
                           'ext': ext_list, 'gvy': gvy_list, 'gvx': gvx_list, 'eoy': eoy_list, 'eox': eox_list,
                           'dmz': dmz_list, 'lcy': lcy_list, 'lcx': lcx_list, 'lcz': lcz_list, 'fbg': fbg_list,
                           'tim': tim_list, 'tid': tid_list, 'act': act_list})


if __name__ == '__main__':
    mode = 0
    if mode == 0:
        folder_name1 = r'/Users/WangHao/Desktop/未命名文件夹'
        path1 = r'/Users/WangHao/Desktop/未命名文件夹'
        if not os.path.exists(path1):
            os.makedirs(path1)
        data_list1, file_list1, num1 = read_npy(folder_name1)
        for i, j in enumerate(num1):
            data1 = data_list1[-1]
            itr_list1, tic_list1, loc_list1, eco_list1, ecc_list1, efo_list1, efc_list1, sta_list1, cfr_list1, \
            dcr_list1, ext_list1, gvy_list1, gvx_list1, eoy_list1, eox_list1, dmz_list1, lcy_list1, lcx_list1, \
            lcz_list1, fbg_list1, tim_list1, tid_list1, act_list1 = get_data(data=data1, dim=3)
            save_path1 = os.path.join(path1, r'{}.mat'.format(file_list1[j]))
            save_mat(save_path1, itr_list1, tic_list1, loc_list1, eco_list1, ecc_list1, efo_list1, efc_list1, sta_list1,
                     cfr_list1, dcr_list1, ext_list1, gvy_list1, gvx_list1, eoy_list1, eox_list1, dmz_list1, lcy_list1,
                     lcx_list1, lcz_list1, fbg_list1, tim_list1, tid_list1, act_list1)

    elif mode == 1:
        # Import the saved data.
        # during the final iteration (denoted by the index -1) of valid events.
        mfx = np.load('\桌面\PSD95 647 RIM 594 M2.npy')
        tic1 = mfx[mfx['vld']]['itr']['tic'][:, -1]
        loc1 = mfx[mfx['vld']]['itr']['loc'][:, -1, :]
        efo1 = mfx[mfx['vld']]['itr']['efo'][:, -1]
        dcr1 = mfx[mfx['vld']]['itr']['dcr'][:, -1]

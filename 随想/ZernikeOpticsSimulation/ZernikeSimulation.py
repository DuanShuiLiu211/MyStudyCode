import os
import time
import numpy as np
import matplotlib.pyplot as plt
from ZernikeOptics import ZernikeWavefront, PsfGenerator3D

if __name__ == '__main__':
    zernike_nm = {
        'astig_vert': (2, -2),
        'astig_obli': (2, 2),
        'coma_vert': (3, -1),
        'coma_obli': (3, 1),
        'spher_1st': (4, 0),
        'spher_2st': (6, 0),
        'spher_3st': (8, 0),
        'trefo_vert': (3, -3),
        'trefo_obli': (3, 3),
    }

    all_name = zernike_nm.keys()

    # -------------------- 模式1：最大最小振幅范围随机均匀采样生成 Psf; 模式2：生成最大最小振幅的 Psf -------------------- #
    mode = 1
    # --------------------------------------------------------------------------------------------------------- #

    file_db1 = 'E:/像差补偿/ZernikeOpticsSimulation/Data/Aberration'
    if not os.path.exists(file_db1):
        os.makedirs(file_db1)
    file_db2 = 'E:/像差补偿/ZernikeOpticsSimulation/Data/Psf'
    if not os.path.exists(file_db2):
        os.makedirs(file_db2)

    if mode == 1:
        nm_amp = {v: np.random.uniform(-0.3875, 0.3875, 1550) for k, v in zernike_nm.items()}
        start1 = time.time()
        for i, (key, value) in enumerate(nm_amp.items()):
            name = list(all_name)[i]
            start2 = time.time()
            for j, v in enumerate(value):
                zwf = ZernikeWavefront({key: v}, order='ansi')
                psf = PsfGenerator3D(psf_shape=(101, 256, 256), units=(0.032, 0.016, 0.016),
                                     na_detection=1.4, lam_detection=0.775, n=1.4)

                zwf_xy = zwf.polynomial(256, outside=0)
                psf_zxy = psf.incoherent_psf(zwf, normed=True)

                indexnews = list(zwf.zernikes.keys())

                zwf_xy_save_dir_base = os.path.join(file_db1, name)
                if not os.path.exists(zwf_xy_save_dir_base):
                    os.makedirs(zwf_xy_save_dir_base)
                zwf_xy_save_dir = os.path.join(zwf_xy_save_dir_base,
                                               f'{name}_{v:.4f}_{indexnews[0].index_noll}_{j}_zwf_xy')
                np.save(zwf_xy_save_dir, zwf_xy)

                psf_zxy_save_dir_base = os.path.join(file_db2, name)
                if not os.path.exists(psf_zxy_save_dir_base):
                    os.makedirs(psf_zxy_save_dir_base)
                psf_zxy_save_dir = os.path.join(psf_zxy_save_dir_base,
                                                f'{name}_{v:.4f}_{indexnews[0].index_noll}_{j}_psf_zxy')
                np.save(psf_zxy_save_dir, psf_zxy)

            end2 = time.time()
            print("运行时间:%.2f秒" % (end2 - start2))
        end1 = time.time()
        print("运行时间:%.2f秒" % (end1 - start1))

    elif mode == 2:
        nm_amp = {v: [-0.3875, 0.3875] for k, v in zernike_nm.items()}
        start1 = time.time()
        for i, (key, value) in enumerate(nm_amp.items()):
            name = list(all_name)[i]
            start2 = time.time()
            for j, v in enumerate(value):
                zwf = ZernikeWavefront({key: v}, order='ansi')
                psf = PsfGenerator3D(psf_shape=(101, 256, 256), units=(0.032, 0.016, 0.016),
                                     na_detection=1.4, lam_detection=0.775, n=1.4)

                zwf_xy = zwf.polynomial(256, outside=0)
                psf_zxy = psf.incoherent_psf(zwf, normed=True)

                indexnews = list(zwf.zernikes.keys())

                zwf_xy_save_dir_base = os.path.join(file_db1,
                                                    f'{name}_'
                                                    f'{indexnews[0].n, indexnews[0].m}_'
                                                    f'{indexnews[0].index_noll}_'
                                                    f'{indexnews[0].index_ansi}')
                if not os.path.exists(zwf_xy_save_dir_base):
                    os.makedirs(zwf_xy_save_dir_base)
                zwf_xy_save_dir = os.path.join(zwf_xy_save_dir_base, f'{name}_{v:.4f}_zwf_xy')

                np.save(zwf_xy_save_dir, zwf_xy)
                plt.imsave(f'{zwf_xy_save_dir}.png', zwf_xy, dpi=300)

                psf_zxy_save_dir_base = os.path.join(file_db2,
                                                     f'{name}_'
                                                     f'{indexnews[0].n, indexnews[0].m}_'
                                                     f'{indexnews[0].index_noll}_'
                                                     f'{indexnews[0].index_ansi}')
                if not os.path.exists(psf_zxy_save_dir_base):
                    os.makedirs(psf_zxy_save_dir_base)
                psf_zxy_save_dir = os.path.join(psf_zxy_save_dir_base, f'{name}_{v:.4f}_psf_zxy')

                np.save(psf_zxy_save_dir, psf_zxy)
                for k, img in enumerate(psf_zxy):
                    plt.imsave(f'{psf_zxy_save_dir}_{k}.png', img, dpi=300)

            end2 = time.time()
            print("运行时间:%.2f秒" % (end2 - start2))
        end1 = time.time()
        print("运行时间:%.2f秒" % (end1 - start1))

import time
import warnings
import numpy as np
from scipy.special import binom
from functools import lru_cache
import matplotlib.pyplot as plt


# first: define zernike function
def nm_polynomial(n, m, rho, theta, normed=True):
    """
    returns the zernike polynomial by classical n,m enumeration

    if normed=True, then they form an orthonormal system

        where each mode has an integral of 1 remark by wanghao

        and the first modes are

        z_nm(0,0)  = 1/sqrt(pi)* 1
        z_nm(1,-1) = 1/sqrt(pi)* 2r cos(phi)
        z_nm(1,1)  = 1/sqrt(pi)* 2r sin(phi)
        z_nm(2,0)  = 1/sqrt(pi)* sqrt(3)(2 r^2 - 1)
        ...
        z_nm(4,0)  = 1/sqrt(pi)* sqrt(5)(6 r^4 - 6 r^2 +1)
        ...

    if normed =False, then they follow the Born/Wolf convention (i.e. min/max is always -1/1)

        no coefficient in the radial part zernike polynomial remark by wanghao

        z_nm(0,0)  = 1
        z_nm(1,-1) = r cos(phi)
        z_nm(1,1)  = r sin(phi)
        z_nm(2,0)  = (2 r^2 - 1)
        ...
        z_nm(4,0)  = (6 r^4 - 6 r^2 +1)
        ...
    """
    if abs(m) > n:  # 第一个条件，值域：n >= |m| >= 0 才存在 zernike 多项式
        raise ValueError(" |m| !<= n, ( %s !<= %s)" % (abs(m), n))

    if (n - abs(m)) % 2 == 1:  # 第二个条件，取值：n - |m| 为奇数则径向多项式为 0
        return 0 * rho + 0 * theta

    radial = 0
    ml = abs(m)

    # zernike 多项式的径向多项式
    for k in range((n - ml) // 2 + 1):
        radial += (-1.) ** k * binom(n - k, k) * binom(n - 2 * k, (n - ml) // 2 - k) * rho ** (n - 2 * k)

    radial *= (rho <= 1.)  # 第三个条件，取值：径向距离 0 <= rho <= 1

    if normed:
        prefac = 1. / np.sqrt((1. + (m == 0)) / (2. * n + 2)) / np.sqrt(np.pi)
    else:
        prefac = 1.

    # normed |zernike| <= 1 / sqrt(pi) or |zernike| <= 1
    if m >= 0:
        return prefac * radial * np.cos(ml * theta)
    else:
        return prefac * radial * np.sin(ml * theta)


# 构建栅格化坐标
@lru_cache(maxsize=256)  # 将耗时的函数结果保存到内存里，函数传入相同的参数无需重复计算
def rho_theta(size):
    dy = np.linspace(-1, 1, size)
    dx = np.linspace(-1, 1, size)
    y, x = np.meshgrid(dy, dx, indexing='ij')  # 关于x轴对称的笛卡尔坐标系
    rho = np.hypot(y, x)  # 直角坐标系求直角三角形斜边，均等分为极径
    theta = np.arctan2(y, x)  # 求正切角（对边：args1，邻边：args2）得极角
    return rho, theta


@lru_cache(maxsize=256)
def outside_mask(size):
    rho, theta = rho_theta(size)
    return nm_polynomial(0, 0, rho, theta, normed=False) < 1  # 返回相差单位圆外坐标


# second: define zernike model
def nm_to_noll(n, m):
    j = (n * (n + 1)) // 2 + abs(m)
    if m > 0 and n % 4 in (0, 1):
        return j
    if m < 0 and n % 4 in (2, 3):
        return j
    if m >= 0 and n % 4 in (2, 3):
        return j + 1
    if m <= 0 and n % 4 in (0, 1):
        return j + 1
    assert False


def nm_to_ansi(n, m):
    return (n * (n + 2) + m) // 2


def present(e):
    if isinstance(e, BaseException):
        raise e
    else:
        raise ValueError(e)


# 计算标准的 zernike 多项式
class Zernike:
    """
        Encapsulates zernike polynomials

        :param index: string, integer or tuple, index of Zernike polynomial e.g. 'defocus', 4, (2, 2)
        :param order: string, defines the Zernike nomenclature if index is an integer, eg noll or ansi, default is noll
    """
    # 常见 zernike 多项式的名称列表
    _ansi_names = ['piston', 'tilt', 'tip', 'oblique astigmatism', 'defocus',
                   'vertical astigmatism', 'vertical trefoil', 'vertical coma',
                   'horizontal coma', 'oblique trefoil', 'oblique quadrafoil',
                   'oblique secondary astigmatism', 'primary spherical',
                   'vertical secondary astigmatism', 'vertical quadrafoil']
    _nm_pairs = set((n, m) for n in range(200) for m in range(-n, n + 1, 2))  # 各种 zernike 多项式指数构成的集合
    _noll_to_nm = dict(zip((nm_to_noll(*nm) for nm in _nm_pairs), _nm_pairs))  # noll 索引：zernike 多项式指数构成的的字典
    _ansi_to_nm = dict(zip((nm_to_ansi(*nm) for nm in _nm_pairs), _nm_pairs))  # ansi 索引：zernike 多项式指数构成的的字典

    def __init__(self, index, order='noll'):
        super().__setattr__('_mutable', True)
        if isinstance(index, str):
            if index.isdigit():
                index = int(index)
            else:
                name = index.lower()
                name in self._ansi_names or present(
                    ValueError("Your input for index is string : Could not identify the name of Zernike polynomial"))
                index = self._ansi_names.index(name)
                order = 'ansi'

        if isinstance(index, (list, tuple)) and len(index) == 2:  # 输入 zernike 多项式指数的元组或者列表索引 zernike 多项式
            self.n, self.m = int(index[0]), int(index[1])
            (self.n, self.m) in self._nm_pairs or present(ValueError(
                "Your input for index is list/tuple : Could not identify the n,m order of Zernike polynomial"))
        elif isinstance(index, int):  # 确保相应索引模式下 zernike 多项式的索引值为整数
            order = str(order).lower()
            order in ('noll', 'ansi') or present(
                ValueError("Your input for index is int : Could not identify the Zernike nomenclature/order"))
            if order == 'noll':
                index in self._noll_to_nm or present(ValueError(
                    "Your input for index is int and input for Zernike nomenclature is Noll:"
                    " Could not identify the Zernike polynomial with this index"))
                self.n, self.m = self._noll_to_nm[index]
            elif order == 'ansi':
                index in self._ansi_to_nm or present(ValueError(
                    "Your input for index is int and input for Zernike nomenclature is ANSI:"
                    " Could not identify the Zernike polynomial with this index"))
                self.n, self.m = self._ansi_to_nm[index]
        else:
            raise ValueError("Could not identify your index input, we accept strings, lists and tuples only")

        self.index_noll = nm_to_noll(self.n, self.m)
        self.index_ansi = nm_to_ansi(self.n, self.m)
        self.name = self._ansi_names[self.index_ansi] if self.index_ansi < len(self._ansi_names) else None
        self._mutable = False

    # 给定栅格数使用 self.phase 方法计算 zernike 多项式的方法，单位圆外默认用 np.nan 填补或 给定的非 np.nan
    def polynomial(self, size, normed=True, outside=np.nan):
        """
            For visualization of Zernike polynomial on a disc of unit radius

            :param size: integer, Defines the shape of square grid, e.g. 256 or 512
            :param normed: boolean, Whether the Zernike polynomials are normalized, default is True
            :param outside: scalar, Outside padding of the spherical disc defined within a square grid, default np.nan
            :return: 2D array, Zernike polynomial computed on a disc of unit radius defined within a square grid
        """
        np.isscalar(size) and int(size) > 0 or present(ValueError())
        return self.phase(*rho_theta(int(size)), normed=normed, outside=outside)

    # 给定极径与极角的计算 zernike 多项式的方法，单位圆外默认用 None 填补或 给定的非 None
    def phase(self, rho, theta, normed=True, outside=None):
        """
            For creation of a zernike polynomial with a given polar co-ordinate system

            :param rho: 2D square array, radial axis
            :param theta: 2D square array, azimuthal axis
            :param normed: boolen, whether the Zernike polynomials are normalized, default is True
            :param outside: scalar, outside padding of the spherical disc defined within a square grid, default is None
            :return: 2D array, Zernike polynomial computed for rho and theta
        """
        isinstance(normed, bool) or present(ValueError('Only boolen flag is accepted'))
        outside is None or np.isscalar(outside) or present(
            ValueError("Only scalar constant value for outside is accepted"))
        ans = nm_polynomial(self.n, self.m, rho, theta, normed=bool(normed))
        if outside is not None:
            ans[nm_polynomial(0, 0, rho, theta, normed=False) < 1] = outside

        return ans

    def __hash__(self):
        return hash((self.n, self.m))

    def __eq__(self, other):
        return isinstance(other, Zernike) and (self.n, self.m) == (other.n, other.m)

    def __lt__(self, other):
        return self.index_ansi < other.index_ansi

    def __setattr__(self, *args):
        if self._mutable:
            super().__setattr__(*args)
        else:
            raise AttributeError('Zernike is immutable')

    def __repr__(self):
        return f'Zernike(n={self.n}, m={self.m: 1}, noll={self.index_noll:2}, ansi={self.index_ansi:2}' + (
            f", name='{self.name}')" if self.name is not None else ")")


# third: combination zernike mode application
# 结构化为索引值：输入振幅的字典
def ensure_dict(values, order='noll'):
    if isinstance(values, dict):
        return values  # 字典数据无需处理
    if isinstance(values, np.ndarray):
        values = tuple(values.ravel())  # 将值拉成一维
    if isinstance(values, (tuple, list)):  # 确认值是元组或者列表
        order = str(order).lower()
        order in ('noll', 'ansi') or present(ValueError("Could not identify the Zernike nomenclature/order"))
        offset = 1 if order == 'noll' else 0
        indices = range(offset, offset + len(values))
        return dict(zip(indices, values))  # 把索引值和振幅值聚合成字典
    raise ValueError("Could not identify the data type for dictionary formation")


# 字典中获取值构成振幅列表
def dict_to_list(kv):
    max_key = max(kv.keys())
    out = [0] * (max_key + 1)
    for k, v in kv.items():
        out[k] = v
    return out


# 计算给定振幅的 zernike 多项式，振幅不能是标量，默认索引模式 noll
class ZernikeWavefront:
    """
        Encapsulates the wavefront defined by zernike polynomials

        :param amplitudes: dictionary, nd array, tuple or list, Amplitudes of Zernike polynomials
        :param order: string, Zernike nomenclature, eg noll or ansi, default is noll
    """
    def __init__(self, amplitudes, order='noll'):
        amplitudes = ensure_dict(amplitudes, order)
        all(np.isscalar(a) for a in amplitudes.values()) or present(
            ValueError("Could not identify scalar value for amplitudes after making a dictionary"))

        self.zernikes = {Zernike(j, order=order): a for j, a in amplitudes.items()}  # 生成包含多模式的 zernike 多项式的字典
        self.amplitudes_noll = tuple(dict_to_list({z.index_noll: a for z, a in self.zernikes.items()})[1:])
        self.amplitudes_ansi = tuple(dict_to_list({z.index_ansi: a for z, a in self.zernikes.items()}))
        self.amplitudes_requested = tuple(self.zernikes[k] for k in sorted(self.zernikes.keys()))

    def __len__(self):
        return len(self.zernikes)

    def polynomial(self, size, normed=True, outside=np.nan):
        """
            For visualization of weighted sum of zernike polynomials on a disc of unit radius

            :param size: integer, Defines the shape of square grid, e.g. 64 or 128 or 256 or 512
            :param normed: boolean, Whether the Zernike polynomials are normalized, default is True
            :param outside: scalar, Outside padding of the spherical disc defined within a square grid,
                                    default is np.nan
            :return: 2D array, weighted sums of Zernike polynomials computed on a disc of unit radius defined
                               within a square grid
        """
        return np.sum([a * z.polynomial(size=size, normed=normed, outside=outside) for z, a in self.zernikes.items()],
                      axis=0)

    def phase(self, rho, theta, normed=True, outside=None):
        """
            For creation of phase defined as a weighted sum of Zernike polynomial with a given polar co-ordinate system

            :param rho: 2D square array,  radial axis
            :param theta: 2D square array, azimuthal axis
            :param normed: boolean, whether the Zernike polynomials are normalized, default is True
            :param outside: scalar, outside padding of the spherical disc defined within a square grid, default is none
            :return: 2D array, wavefront computed for rho and theta
        """
        return np.sum(
            [a * z.phase(rho=rho, theta=theta, normed=normed, outside=outside) for z, a in self.zernikes.items()],
            axis=0)


# 给定振幅范围并随机产生振幅，计算 zernike 多项式
def random_zernike_wavefront(amplitude_ranges, order='noll'):
    """
        Creates random zernike wavefront with random amplitudes drawn from a uniform distribution

        :param amplitude_ranges: dictionary, nd array, tuple or list, amplitude bounds
        :param order: string, to define the Zernike nomenclature if index is an integer, eg noll or ansi,
                              default is noll
        :return: Zernike wavefront object
    """
    ranges = np.random
    amplitude_ranges = ensure_dict(amplitude_ranges, order)
    all((np.isscalar(v) and v >= 0) or (isinstance(v, (tuple, list)) and len(v) == 2) for v in
        amplitude_ranges.values()) or present(ValueError('false in one elements of the iterable'))  # 必须全部跌倒迭代都正确
    amplitude_ranges = {k: ((-v, v) if np.isscalar(v) else v) for k, v in amplitude_ranges.items()}
    all(v[0] <= v[1] for v in amplitude_ranges.values()) or present(
        ValueError("Lower bound is expected to be less than the upper bound"))
    return ZernikeWavefront({k: ranges.uniform(*v) for k, v in amplitude_ranges.items()}, order=order)


class PsfGenerator3D:
    def __init__(self, psf_shape, units, lam_detection, n, na_detection):

        """
        Encapsulates 3d psf generator

        :param psf_shape: tuple, psf shape as (z, y, x), e.g. (64, 64, 64)
        :param units: tuple, voxel size in microns, e.g. (0.1, 0.1, 0.1)
        :param lam_detection: scalar, wavelength in microns, e.g. 0.632
        :param n: scalar, refractive index, eg 1.33
        :param na_detection: scalar, numerical aperture of detection objective, eg 1.4
        """
        psf_shape = tuple(psf_shape)
        units = tuple(units)
        self.na_detection = na_detection
        self.n = n
        self.lam_detection = lam_detection
        self.nz, self.ny, self.nx = psf_shape
        self.dz, self.dy, self.dx = units

        # 生成频域的频谱坐标 (kx, ky) 其中 (dkx, dky) = (1 / nx*dx, 1 / ny*dy) 相应空域的分辨率为 (dx, dy)
        # f = [0, 1, ..., n / 2 - 1, -n / 2, ..., -1] / (d * n) if n is even
        # f = [0, 1, ..., (n - 1) / 2, -(n - 1) / 2, ..., -1] / (d * n) if n is odd
        # (max(kx), max(ky)) = (((nx // 2) + 1) / (nx * dx), ((ny // 2) + 1) / (ny * dy))
        ky = np.fft.fftfreq(self.ny, self.dy)
        kx = np.fft.fftfreq(self.nx, self.dx)

        # 生成 z 关于焦平面对称分辨率为 dz
        if self.nz % 2 == 0:
            z = self.dz * (np.arange(self.nz) - (self.nz - 1) / 2)
        else:
            z = self.dz * (np.arange(self.nz) - self.nz // 2)

        # z 个 xoy 频域坐标中的理想光场：p(kx, ky) * exp(-j2πz/λ * sqrt((n/λ)^2 - ((kx/λ)^2 + (ky/λ)^2)))
        self.kz3, self.ky3, self.kx3 = np.meshgrid(z, ky, kx, indexing="ij")

        self.k_cut = 1. * na_detection / self.lam_detection  # 截止频率
        kr3 = np.sqrt(self.ky3 ** 2 + self.kx3 ** 2)
        self.k_mask3 = (kr3 <= self.k_cut)  # pupil function：p(kx, ky) = 1, (na/λ)^2 => (kx^2 + ky^2)

        warnings.filterwarnings("ignore")
        self.h = np.sqrt(1. * self.n ** 2 - kr3 ** 2 * lam_detection ** 2)
        self.k_prop = np.exp(-2.j * np.pi * self.kz3 / lam_detection * self.h)
        self.k_prop[np.isnan(self.h)] = 0.
        self.k_base = self.k_mask3 * self.k_prop

        # xoy 像差的频域极坐标
        ky2, kx2 = np.meshgrid(ky, kx, indexing="ij")
        kr2 = np.hypot(ky2, kx2)

        self.k_rho = kr2 / self.k_cut  # 调整像差单位圆内的频谱密度,相应调整空域视场,至关重要
        self.k_phi = np.arctan2(ky2, kx2)
        self.k_mask2 = (kr2 <= self.k_cut)

        # xoy 频域的傅里叶逆变换
        self.my_ifftn = lambda x: np.fft.ifftn(x, axes=(1, 2))

    def masked_phase(self, phi, normed=True):
        """
        Returns masked zernike polynomial for back focal plane, masked according to the setup

        :param phi: Zernike/ZernikeWavefront object
        :param normed: boolean, multiplied by normalization factor, eg True
        :return: masked wavefront, 2d array
        """
        return self.k_mask2 * phi.phase(self.k_rho, self.k_phi, normed=normed, outside=None)

    def coherent_psf(self, phi, normed=True):
        """
        Returns the coherent psf for a given wavefront phi and no shift

        :param phi: Zernike/ZernikeWavefront object
        :param normed: boolean, multiplied by normalization factor, eg True
        :return: coherent psf, 3d array
        """
        # 引入像差后的光场：p(kx, ky) * exp(-j2πz * sqrt((n/λ)^2 - (kx^2 + ky^2))) * exp(-j2π * φ(kx, ky)/λ)
        phi = self.masked_phase(phi, normed=normed)
        ku = self.k_base * np.exp(2.j * np.pi * phi / self.lam_detection)
        return self.my_ifftn(ku)

    def incoherent_psf(self, phi, normed=True):
        """
        Returns the incoherent psf for a given wavefront phi
           (which is just the squared absolute value of the coherent one)
           The psf is normalized such that the sum intensity on each plane equals one

        :param phi: Zernike/ZernikeWavefront object
        :param normed: boolean, multiplied by normalization factor, eg True
        :return: incoherent psf, 3d array
        """
        psf = np.abs(self.coherent_psf(phi, normed=normed)) ** 2  # 取模的平方
        psf = np.array([p / np.sum(p) for p in psf])  # 每个面的总能量几乎是相同的，使其总和为1
        return np.fft.fftshift(psf, axes=(1, 2))  # yx 轴都交换半空间


if __name__ == '__main__':
    mode = 2
    if mode == 1:
        f1 = Zernike((1, 1), order='ansi')
        aberration1 = f1.polynomial(512)
        plt.imshow(aberration1)
        plt.colorbar()
        plt.axis('off')
        plt.show()

        amp = np.random.uniform(-1, 1, 4)
        f2 = ZernikeWavefront(amp, order='ansi')
        aberration2 = f2.polynomial(512)
        plt.imshow(aberration2)
        plt.colorbar()
        plt.axis('off')
        plt.show()

        f3 = random_zernike_wavefront([(0, 0), (-1, 1), (1, 2)], order='ansi')
        aberration3 = f3.polynomial(512)
        plt.imshow(aberration3)
        plt.colorbar()
        plt.axis('off')
        plt.show()

    elif mode == 2:
        start = time.time()
        psf1 = PsfGenerator3D(psf_shape=(95, 256, 256), units=(0.032, 0.016, 0.016),
                              na_detection=1.4, lam_detection=0.775, n=1.4)
        wf1 = ZernikeWavefront({(3, -3): 0.3875}, order='ansi')
        h1 = psf1.incoherent_psf(wf1, normed=True)
        for idx, img in enumerate(h1):
            print(idx, np.max(img))
            print(idx, img.sum())
            plt.imsave(f'/Users/WangHao/Desktop/1/{idx}.png', img, dpi=300)
        end = time.time()
        print("运行时间:%.2f秒" % (end - start))
        w1 = wf1.polynomial(64)
        phase1 = wf1.phase(psf1.k_rho, psf1.k_phi, normed=True, outside=None)
        phase1 = np.fft.fftshift(phase1)

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 3, 1)
        plt.imshow(w1, cmap="hot")
        plt.title('Aberration_xy')
        plt.colorbar()

        plt.subplot(2, 3, 2)
        plt.imshow(phase1, cmap="hot")
        plt.title('Aberration_ifftkxky')
        plt.colorbar()

        plt.subplot(2, 3, 4)
        plt.imshow(h1[h1.shape[0] // 2], cmap="hot")
        plt.title('psf_xoy')
        plt.colorbar()

        plt.subplot(2, 3, 5)
        plt.imshow(h1[:, h1.shape[1] // 2, :], cmap="hot")
        plt.title('psf_xoz')
        plt.colorbar()

        plt.subplot(2, 3, 6)
        plt.imshow(h1[:, :, h1.shape[2] // 2], cmap="hot")
        plt.title('psf_yoz')
        plt.colorbar()
        plt.show()

    elif mode == 3:
        ky1 = np.fft.fftfreq(256, 0.016)
        kx1 = np.fft.fftfreq(256, 0.016)
        y1, x1 = np.meshgrid(ky1, kx1, indexing="ij")
        r = np.hypot(y1, x1)
        k_cut = 1. * 1.4 / 0.775
        k_rho = r / k_cut
        k_phi = np.arctan2(y1, x1)
        k_mask = (r <= k_cut)
        f4 = ZernikeWavefront({(3, -3): 0.3875}, order='ansi')

        ab1 = np.fft.fftshift(k_mask * f4.phase(k_rho, k_phi, normed=True, outside=None))
        ab12 = np.fft.fftshift(np.abs(np.fft.ifftn(ab1, axes=(0, 1))) ** 2)
        ab2 = np.fft.fftshift(k_mask * f4.phase(r, k_phi, normed=True, outside=None))
        ab22 = np.fft.fftshift(np.abs(np.fft.ifftn(ab2, axes=(0, 1))) ** 2)
        ab3 = np.fft.fftshift(f4.phase(k_rho, k_phi, normed=True, outside=None))
        ab32 = np.fft.fftshift(np.abs(np.fft.ifftn(ab3, axes=(0, 1))) ** 2)
        aberration4 = f4.polynomial(256, normed=True, outside=None)

        print(np.max(ab1), ab1.sum(), np.max(ab2), ab2.sum(), np.max(ab3), ab3.sum())
        print(np.max(ab12), ab12.sum(), np.max(ab22), ab22.sum(), np.max(ab32), ab32.sum())
        print(np.max(aberration4), aberration4.sum())
        print(np.abs((ab32 - ab12)).sum())
        print(np.abs((ab32 - ab22)).sum())
        print(np.abs((ab22 - ab12)).sum())

        fig = plt.figure(figsize=(20, 8))
        plt.subplot(2, 4, 1)
        plt.imshow(ab1, cmap="hot")
        plt.colorbar()
        plt.title('fft')

        plt.subplot(2, 4, 5)
        plt.imshow(ab12, cmap="hot")
        plt.colorbar()
        plt.title('ifft')

        plt.subplot(2, 4, 2)
        plt.imshow(ab2, cmap="hot")
        plt.colorbar()
        plt.title('fft_no_cut')

        plt.subplot(2, 4, 6)
        plt.imshow(ab22, cmap="hot")
        plt.colorbar()
        plt.title('ifft_no_cut')

        plt.subplot(2, 4, 3)
        plt.imshow(ab3, cmap="hot")
        plt.colorbar()
        plt.title('ifft_no_mask')

        plt.subplot(2, 4, 7)
        plt.imshow(ab32, cmap="hot")
        plt.colorbar()
        plt.title('ifft_no_cut')

        fig.add_subplot(2, 4, (4, 8))
        plt.imshow(aberration4, cmap="hot")
        plt.colorbar()
        plt.title('aberration')

        plt.show()

import numpy as np
import matplotlib.pyplot as plot

# 举例一
'''
figure(num=None, figsize=None, dpi=None, facecolor=None, 
       edgecolor=None, frameon=True, FigureClass=Figure, 
       clear=False, **kwargs):
'''
plot.figure(1, figsize=(4, 4), dpi=300, frameon=True)
ax1 = plot.subplot(2, 2, 1)
# 添加没有框架的子图
ax2 = plot.subplot(222, frameon=False)
# 添加极坐标子图
ax3 = plot.subplot(223, projection='polar')
# 共享坐标轴
ax4 = plot.subplot(224, sharex=ax1, facecolor='red')
# 删除坐标轴
plot.delaxes(ax2)
'''
savefig(fname, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
'''
plot.savefig('image1.png')
plot.show()
plot.figure(num=1, clear=True)
plot.close(plot.figure(1))

# 举例二
fig = plot.figure(2, figsize=(2, 2), dpi=300, frameon=True)
image = np.array([[1, 1, 1, 1],[2, 2, 2 ,2],[3, 3, 3, 3],[4, 4, 4, 4]])
# 绘图时可以设置绘图框架轴的横纵比，色彩图以及色彩映射关系，插值方法和绘图区域位置等
'''
imshow(X, cmap=None, norm=None, aspect=None,
       interpolation=None, alpha=None, vmin=None, vmax=None,
       origin=None, extent=None, *, filternorm=True, filterrad=4.0,
       resample=None, url=None, **kwargs)
'''
plot.imshow(image, cmap='gray')
plot.margins(0., 0.)
plot.axis('off')
plot.subplots_adjust(0.,0.,1.,1.,0.,0.)
plot.close(plot.figure(2))
fig.savefig('image2.tiff', bbox_inches='tight', pad_inches=0., transparent=True)
plot.show()


# 举例三
'''
if format is png:
_pil_png_to_float_array()  # raw norm to 0~1 ndarray
else:
pil_to_array()  # raw to ndarray
'''
data = plot.imread('W:\Study Flies\PyCharm\Script\随想\image2.tiff')
'''
imsave(fname, arr, vmin=None, vmax=None, cmap=None, format=None,
       origin=None, dpi=100, *, metadata=None, pil_kwargs=None):
'''
plot.imsave('image3.png', data, vmin=np.min(data), vmax=np.max(data))
data1 = plot.imread('W:\Study Flies\PyCharm\Script\随想\image3.png')

pass
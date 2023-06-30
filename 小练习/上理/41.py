import tifffile


img = tifffile.imread(
    "/Users/WangHao/Desktop/Paper ImT UNet/data_0913/MT/N1_LD/sequence-as-stack-MT1.N1.LD-BP-250.tif"
)
k = [i for i in range(len(img)) if i % 5 == 0]
img = img[k, ...]
tifffile.imsave(
    "/Users/WangHao/Desktop/Paper ImT UNet/data_0913/MT/N1_LD/sequence-as-stack-MT1.N1.LD-BP-250_r5000.tif",
    img
)

print('运行完成')

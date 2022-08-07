import pandas as pd
import os

path = r'/Users/WangHao/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/280e1315d670b6c1a758e4ce0920bab2/Message/MessageTemp/0ce3e26a3d08f296a46ba353824969d6/File/'
file_list = os.listdir(path)
result = []
for n, name in enumerate(file_list):
    if name[-4::] == '.dat':
        result.append(pd.read_table(path + name, sep='::', header=None, engine='python'))
        a = result[-1][0]
        b = a.values
        c = str(b).replace('\\', '')[2:-2:]
        with open('result.txt', 'a') as f:
            f.write('{}\n'.format(c))

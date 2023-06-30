# encoding:utf-8
import urllib.request as ur
<<<<<<< HEAD
=======
import os
>>>>>>> a98794fef118e4fbd47d0348edb5f8b3154dd000
from urllib.parse import unquote

with open(r"W:\桌面\urls.txt", "r") as fd:
    lines = fd.readlines()
    for line in lines:
        line = line.replace("\n", "")
        filename = line[line.rindex("/") + 1:]
        filename = unquote(filename)
        data = ur.urlopen(line, timeout=20).read()
        with open(filename, "wb") as datafd:
            datafd.write(data)
            datafd.close()
        print(filename)

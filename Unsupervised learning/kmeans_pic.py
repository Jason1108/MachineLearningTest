#coding:utf8

import PIL.Image as image

file_path = 'data/bull.jpg'
f = open(file_path, 'rb')
data = []
img = image.open(f)
m, n = img.size
for i in range(m):
    for j in range(n):
        x, y, z = img.getpixel((i, j))
        data.append([x / 256.0, y / 256.0, z / 256.0])
f.close()

# to be continue..
# Importing Image class from PIL module
from PIL import Image, ImageOps
import glob
import os
import hashlib

inpt = "raw"
outpt = "general"


size = 32

hashs = []

counter = 0

if os.path.exists("../datasets/"+outpt) == False:
    os.mkdir("../datasets/"+outpt)

def crop(im, left, top): 
    right = left+size
    bottom = top+size
    im = im.crop((left, top, right, bottom))
    pixdata = im.load()
    width, height = im.size
    for y in range(height):
        for x in range(width):
            if pixdata[x, y][0] != 0:
                pixdata[x, y] = (255, 255, 255, 0) # remove purple line 
            if pixdata[x, y] == (255, 255, 255, 255):
                pixdata[x, y] = (255, 255, 255, 0) # make transparent
    im = im.crop(im.getbbox())
    width, height = im.size
    im1 = Image.new('L', (size, size), "white")
    w = int((size - width) / 2)
    h = int((size - height) / 2)
    im1.paste(im, (w, h))
    return im1

def save(im):
    global counter
    md5hash = hashlib.md5(im.tobytes()).hexdigest()
    if md5hash in hashs:
        return
    hashs.append(md5hash)
    im.save("../datasets/"+outpt+"/"+str(counter)+".png")
    im = ImageOps.mirror(im)
    counter += 1
    im.save("../datasets/"+outpt+"/"+str(counter)+".png")
    counter += 1

for file in glob.glob("../datasets/"+inpt+"/*.png"):
    im = Image.open(file)
    im1 = crop(im, 18, 11)
    im2 = crop(im, 54, 11)
    save(im1)
    save(im2)
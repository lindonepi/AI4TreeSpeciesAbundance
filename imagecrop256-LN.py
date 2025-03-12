import os
import PIL
from PIL import Image
import image_slicer
import random
import time
import cv2
import numpy as np

# ISTRUCTION :
# use this script for crop image  - use this script for train/val dataset
# run this script in dataset/train/ folder

directory = os.getcwd()
speciex=input("enter the name of the tree species")

directoryout=directory+"/output/"

def rimpiccioliscigrandifoto():

 print("riduce foto grande")
 img = Image.open(nomecompleto)
 y = img.size[1]
 x = img.size[0]
 xvar=0
 yvar=0
 if (( y > 00) and ( x > 800)):
        basewidth=450
        wpercent = (basewidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        if ((img.size[0] > basewidth)):
          img = img.resize((basewidth,hsize), Image.Resampling.LANCZOS)
          img.save(nomecompleto)
 






def rimpicciolisci256(imagedacroppare,nomecompletosave):

 print("riduce foto a 256")

 img2 = imagedacroppare
 y = img2.size[1]
 x = img2.size[0]
 xvar=0
 yvar=0
 print(x,y)
 if (( y > 200) and ( x > 200)):
          basewidth=500
          img2 = img2.resize((256,256), Image.Resampling.LANCZOS)
          img2.save(nomecompletosave)



def controllanero(image_slicer):
 totalpixel=256*256
    #gray_image = cv2.cvtColor(image_slicer, cv2.COLOR_BGR2GRAY) 
 numberblack = 0
 numberred = 0
 for pixel in image_slicer.getdata():
    if pixel == (0, 0, 0): # if your image is RGB (if RGBA, (0, 0, 0, 255) or so
        numberblack += 1
    else:
        numberred += 1
 print('black=' + str(numberblack)+', red='+str(numberred))
 percentualenero=(numberblack  *100)/ totalpixel
 print("Percentuale NERO= " , percentualenero)
 if (percentualenero <=10):
       return 1
 else:
       return 0


def eseguicrop():
 img = Image.open(nomecompleto)
 y = img.size[1]
 x = img.size[0]
 xvar=0
 yvar=0
#imposta size minima del ritaglio 
 sizeminima =256
 if (( y > sizeminima) and ( x > sizeminima)):    
  while  xvar < ( x-sizeminima):
   yvar=0
   while yvar < ( y-sizeminima):
    print(xvar,yvar)
    image_slicer=img.crop((xvar,yvar,xvar+sizeminima,yvar+sizeminima)) 
   # funzione controllanero restituisce 1 se percentuale nero minore 10 percento
    if (controllanero(image_slicer)==1):
      print("percentuale minore di 10 --> scrivi immagine")
#     tiles = image_slicer.slice(nomecompleto, 4, save=False)
      casuale= random.randint(332, 402902)
      prefissofoto="slice256-"+ str(casuale)
      #image_slicer.save_tiles(tiles, directory=directory, prefix=prefissofoto)
#      nomecompletosave=directory+"/cropped2/"+prefissofoto+".jpg"
      nomecompletosave=directoryout+prefissofoto+".jpg"
      image_slicer.save(nomecompletosave)
    yvar = yvar+60
   xvar = xvar+60





def exec_crop(f, dirs):
    global directory
    global nomecompleto
    global sottodir
    print(" eseguo crop su " + nomecompleto)
    
    eseguicrop()

# MAIN

print(" EXEC CROP FOR ALL SUBFOLDER -  Root DIRECTORY = " + directory)
time.sleep(4)
for root, dirs, files in os.walk(directory):
    for filename in files:
        print(os.path.join(root, filename))
        nomecompleto=(os.path.join(root, filename))
        print("root "+ root)


        print("filename = " + filename)
    # checking if   size < 400 Kb
        if (os.path.getsize(os.path.join(root, filename)) > 150):
            print(filename)
            nomejpg=speciex+".jpg"
            nomejpeg=speciex+".jpeg"
        if (nomecompleto.endswith(nomejpg) or nomecompleto.endswith(nomejpeg)):
            #   rimpiccioliscigrandifoto()
               exec_crop(filename,root)

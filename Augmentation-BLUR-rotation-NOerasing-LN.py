import Augmentor
import os
import PIL
#import Pillow
#from Pillow import ImageEnhance
from PIL import Image, ImageEnhance, ImageFilter
import random

# use this script for data augmentation . Run script into root folder of train dataset 


path_data= os.getcwd()
print(os.getcwd())
print(" AUGMENTATION SCRIPT  for Color modification  -  BLUR - ROTATE ") 
x=input("press a key ")
count = 0



print("PATH IMMAGINI  =  " + str(path_data))



def augmentatorfile(moltiplica):

	print("inizio AUGM")
	p = Augmentor.Pipeline(path_data)
	#p = Augmentor.Pipeline(path_data,output_directory=path_data)
	#p = Augmentor.Pipeline(path_data,output_directory="/tmp/augmented-v2/")

	#p.crop_random( percentage_area=, probability=0.1)
	#p.crop_by_size(0.9,256,256,False)

	p.rotate(probability=0.9, max_left_rotation=20, max_right_rotation=20)

	p.random_brightness(probability=0.4,    min_factor=.9,    max_factor=1.2)

	#p.zoom(probability=0.3, min_factor=1.1, max_factor=1.2)
	# skewing  - inclinazione da diverse prospettive
	p.skew_tilt(0.2)
	p.random_erasing(probability=0.8, rectangle_area=0.2)
	#p.flip_random(0.3)
	p.flip_left_right(0.9)
	p.flip_top_bottom(0.9)
	#p.sample(10)

	#numeroaugment=count-(round(count/2))
	numeroaugment=count*moltiplica
	print(" genera un numero di immagini pari " + str(numeroaugment))
	p.sample(numeroaugment)


def pilfile():
	with Image.open(nomefile) as img:
	  img.load()
	  print(type(img))
	  print(img.format)
# scala di grigi 
	  gray_img = img.convert("L")
#split rgb  https://realpython.com/image-processing-with-the-python-pillow-library/
# red, green, blue = img.split()

# filter
	if (str(img.format)=="JPEG"):
		print("file JPEG")

		sharp_img = img.filter(ImageFilter.SHARPEN)
		edge_enhance = img.filter(ImageFilter.EDGE_ENHANCE)
		regolacontrasto=ImageEnhance.Contrast(img)
		blur_image=img.filter(ImageFilter.EDGE_ENHANCE)
		nomefilerand="immagine"+str(random.randint(1001,84000))+".jpg"
		nomesharp="sharp-"+nomefilerand
		nomeedge="edge-"+nomefilerand
		nomeblur = "blur-" + nomefilerand
		sharp_img.save(nomesharp)
		edge_enhance.save(nomeedge)
		blur_image.save(nomeblur)

# esegui augmentor su cartella 
#  augmentorfile()


#os.mkdir("./output")


# Iterate directory
print(" STEP 1 :  crea immagini con EDGEenhance  - CONTRAST - SHARP" )
for path in os.listdir(path_data):
    # check if current path is a file
    nomefile=str(os.path.join(path_data, path))
    if os.path.isfile(os.path.join(path_data, path)):
      count += 1
    if (nomefile.lower().find(".jpg") > 0):
      pilfile()
      print("ok")



#CALCOLO TOTALE FIL PRESENTI
print('File count originale :', count)

lst = os.listdir(os.getcwd()) # your directory path
number_files = len(lst)
print (" FILE TOT PRESENTI = " +str(number_files))
moltiplicastr=input("INDICA di quante volte vuoi moltiplicare il numero di file rispetto a file count originale")
moltiplica=int(moltiplicastr)
augmentatorfile(moltiplica)







#from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
#from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# #from pytorch_grad_cam.utils.image import show_cam_on_image


# READ THIS
#Use this script for species abundances automatic calculation 

import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

import torch
from torch import nn
from torchvision import models, transforms
import random
import matplotlib.image as mpimg
import shutil

import csv

random.seed(31)
torch.manual_seed(31)

#os.environ["CUDA_VISIBLE_DEVICES"]="0"

data="270225"
nomezona="Carpegna"
#nomezona="Conero"
#nomezona="Furlo"
#nomezona="Pievebovigliana"
tile_size=224


# GLOBAL VAR



img_pathglobal="/home/lindo/project/AI4TreeSpeciesAbundances/dataset-testing/"+nomezona+"/"
image_path=image_pathglobal=img_pathglobal





# FUNCTIONS



def main_vggresnet(modrete,img_nome,filenamejpg):

    # Opens image from disk, normalizes it and converts to tensor
    read_tensor = transforms.Compose([
        lambda x: Image.open(x),
        lambda x: x.convert('RGB'),
        #transforms.Resize(400),
        #transforms.RandomResizedCrop(256),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        lambda x: torch.unsqueeze(x, 0)
    ])


    class Flatten(nn.Module):
        #One layer module that flattens its input.

        def __init__(self):
            super(Flatten, self).__init__()

        def forward(self, x):
            return x.view(x.size(0), -1)


    # Given label number returns class name
    def get_class_name(c):
    #    labels = [ 'Aceropalusobtusatum','Castaneasativa','Fraxinusornus','Ostryacarpinifolia','Populustremula' ,'Prunusavium','Quercuscerris', 'Quercusilex','Quercuspubescens']
        print("funzione get class name" +str(c))
        #print(' '.join(labels[c].split(',')[0].split()[1:]))
        return ' '.join(labels[c].split(',')[0].split()[1:])
        #return labels

    #Loadmodel

    #device ="cuda:0"
    device ="cpu"

    #modrete=input("scrivi modello : vgg19 o vgg16 o resnet101=") 
    if modrete=="vgg19":
     modello="VGG19"
     model = models.vgg19(pretrained=True)
    #agg 2025 -vgg19
#     model =torch.load('../ultimo-25Luglio24mattonelle-VGG-augmented_agg310125-acc089.pth', weights_only=False)
     model =torch.load('ultimo-test14022025-dataset2025-mattonelle-Res101-augmented.pth', weights_only=False)

    if modrete=="vgg16":
     modello="VGG16"
     model = models.vgg16(pretrained=True)
    #agg 2025 vgg16 
     model = torch.load('ultimo-test28012025-mattonelle-VGG16-input256-augmented.pth',weights_only=False)

    if modrete=="resnet101":
     modello="resnet101"
    #model = models.vgg16(pretrained=True)
    #agg 2025  resnet
     model = torch.load('ultimo-test04022025-mattonelle-Res101-augmented-ACC094.pth',weights_only=False)
    congela="Y"

    for param in model.parameters():
      if (congela=="Y" or congela=="y"):
        print("layer iniziali congelati")
        param.requires_grad = False

    # scongela i layer  dopo un certo numero di epoche
      else:  
        param.requires_grad = True


    labels = [ 'Aceropalusobtusatum','Castaneasativa','Fraxinusornus','Ostryacarpinifolia','Populustremula' ,'Prunusavium','Quercuscerris', 'Quercusilex','Quercuspubescens']

    numeroclassi=9


    model.to(device)


    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())


    model = model.eval()
    model = model.cuda()

    #Grad - CAM

    #Example

    total_samples=0
    total_correct=0

    #creo dizionario per classi e indici
    #classi_list= ["aceropalus","castaneasativa","oleaeuropeae","quercuspubescens" ]
    classi_list = [ 'Aceropalusobtusatum','Castaneasativa','Fraxinusornus','Ostryacarpinifolia','Populustremula' ,'Prunusavium','Quercuscerris', 'Quercusilex','Quercuspubescens']


    dir_path="/tmp/jpg/"
   
    # ciclo per ogni file 
    nomecsv="CSV-dataset2025-rete-"+modrete+"-224NOpad-zona-"+nomezona+data+".csv"
    #with open(nomecsv, mode='w' , encoding='utf-8') as csvfile:
    # csvimage_writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE)
    # csvimage_writer = csv.writer(csvfile , delimiter=',')
    # csvimage_writer = csv.writer(nomecsv)
    # fields=['NameFileInput','nomemodello','size', 'Class-n1', 'Class-n2', 'Class-n3','Class-n4', 'Class-n5', 'Class-n6','Class-n7', 'Class-n8', 'Class-n9','vuoto','Perc-class1','Perc-class2','Perc-class3','Perc-class4','Perc-class5','Perc-class6','Perc-class7','Perc-class8','Perc-class9'] 
    # csvimage_writer.writerow(fields)


    # genera listanomi tile in base a coordinate 
    #nomeimg=input("Inserisci nome img es DJI_00xx senza estensione =")
    listafile=[]
    listaprimaclasse=[]
    #xmax=12
    #ymax=12
    #for x in range(xmax):
    # for  y in range(ymax):
      
    # for ognifile in  listafile:
    for ognifile in sorted(os.listdir(dir_path)):
      img_path=dir_path+ognifile
      print(str(ognifile))
      total_samples = total_samples+1

      img_tensor = read_tensor(img_path)
      pp, cc = torch.topk(nn.Softmax(dim=1)(model(img_tensor.cuda())), numeroclassi)

        #pp, cc = torch.topk(nn.ReLU()(model(img_tensor.cuda())), 4)
      max_class = cc[0][0].item()
      print("CC = " ,cc)
      print( "PP =" ,pp)
      stringacsv=str(ognifile)
      stringaclassi=""
      listacsv=[stringacsv]
    # for i in range(cc[0].item()):
      for i in range(9):
        #stringaclassi=stringaclassi+","+str(cc[0][i].item())

        listacsv.append(str(labels[cc[0][i].item()]))

      listaprimaclasse.append(cc[0][0].item()) 
      listacsv.append(" ")
    # aggiungi percentuali delle classi       
      
    #calcola statistiche abbondanza  e scrivi su file 


    #nome img inputsenza estensione  es dji_0087 
    nomeimg = filenamejpg 
    nomefile=nomezona+"-abbondanza-dataset2024-"+modello+"-immagine-"+nomeimg+"-size224NOpad-210225.txt"

    f = open(nomefile, "w") 

    numelem=len(listaprimaclasse)
    #if numelem==0:
    # numelem=1
    f.write(modello)
    with open(nomecsv, mode='a' , encoding='utf-8', newline='') as csvfile:
     csvimage_writer = csv.writer(csvfile)
     #csvimage_writer.writeheader()
     #csvimage_writer.write(fields)

     for m in range(9):
      res = sum(1 for i in  listaprimaclasse if i ==m)
    # print("classe " +str(labels[m])+  " num mattonelle primaclasse = " +str(res))
      perc=round(100*float(res/numelem),1)
      stringa="classe " +str(labels[m])+  " abbondanza = " +str(perc)
      print(stringa)
      stringa=stringa+"\n"
      f.write(stringa)
      listacsv.append(str(perc))
      #listavitcsv=listavitcsv+";"+str(perc)
      #listavitcsv.append()
     f.close()
      

  #fields=['NomeZona','NomeFileJPG','ACeropalus', 'CastaneaSativa', 'Fraxinus', 'OstryaCarpinifolia','PopulusTremula', 'PrunusAvium', 'QuercusCerris','Quercusilex', 'Quercuspubescens',] 
     csvimage_writer.writerow(listacsv)
     #print("Id classe conteggio  valori STRANI maggiori 9 =" , ppstrani)
     csvfile.close()


#fine script vgg-resnet

# SCRIPT VIT

def test_vit(nomefilejpgvit):

    from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image


    import os
    import numpy as np
    from matplotlib import pyplot as plt
    from PIL import Image

    import torch
    from torch import nn
    from torchvision import models, transforms
    import random
    import matplotlib.image as mpimg
    import shutil

    import csv

    random.seed(31)
    torch.manual_seed(31)


    # Opens image from disk, normalizes it and converts to tensor
    read_tensor = transforms.Compose([
        lambda x: Image.open(x),
        lambda x: x.convert('RGB'),
    # ALERT : NOT USE RANDOMCROP 
 
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        lambda x: torch.unsqueeze(x, 0)
    ])


    class Flatten(nn.Module):
        """One layer module that flattens its input."""

        def __init__(self):
            super(Flatten, self).__init__()

        def forward(self, x):
            return x.view(x.size(0), -1)


    # Given label number returns class name
    def get_class_name(c):
    #    labels = [ 'Aceropalusobtusatum','Castaneasativa','Fraxinusornus','Ostryacarpinifolia','Populustremula' ,'Prunusavium','Quercuscerris', 'Quercusilex','Quercuspubescens']
        print("funzione get class name" +str(c))
        #print(' '.join(labels[c].split(',')[0].split()[1:]))
        return ' '.join(labels[c].split(',')[0].split()[1:])
        #return labels

    #Loadmodel

    #Here, model is splitintotwoparts: featureextractor and classifier.Provided is theimplementationfor ResNet / VGG / DenseNet architechtures.

    #Here, Flattenlayer is beingbuilt in in themodel as well.InPyTorchimplementationflattenning is done in theforwardpass, butweneedit as aseparate layer.



    #device ="cuda:0"
    device="cpu"


    #baseline
    #model = torch.load('ultimo-VIT-BASE-augmented-Pievebovigliana-partial-MATTONELLE-GEN2024.pth',weights_only=False)

    #altri modelli -gen 2025

    model = torch.load('ultimo-VIT-H14-224-Pievebovigliana-dataset2025-MATTONELLE-160225.pth',weights_only=False)
#    model = torch.load('ultimo-VITH14-BASE-Pievebovigliana-dataset2024versionebase-MATTONELLE256-200225.pth',weights_only=False)



    labels = [ 'Aceropalusobtusatum','Castaneasativa','Fraxinusornus','Ostryacarpinifolia','Populustremula' ,'Prunusavium','Quercuscerris', 'Quercusilex','Quercuspubescens']

    #model = models.vgg19(pretrained=True)
    #model = models.resnet50(pretrained=True)
    #model = models.densenet161(num_classes=4)
    #model = models.densenet161(num_classes=3)
    #model = models.densenet161(pretrained=True)


    #model =torch.load('/tmp/ultimo-densenet161.pth')


    #model = torch.load('/tmp/ultimo-densenet161-augmented.pth')


    congela="y"

    for param in model.parameters():
      if (congela=="Y" or congela=="y"):
        print("layer iniziali congelati")
        param.requires_grad = False

    # scongela i layer  dopo un certo numero di epoche
      else:  
        param.requires_grad = True





    numeroclassi=9
    num_ftrs = 8



    print(model)
    model.to(device)


    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())


    model = model.eval()
    model = model.cuda()

    #Grad - CAM


    #Example

    total_samples=0
    total_correct=0

    #creo dizionario per classi e indici
    #classi_list= ["aceropalus","castaneasativa","oleaeuropeae","quercuspubescens" ]
    classi_list = [ 'Aceropalusobtusatum','Castaneasativa','Fraxinusornus','Ostryacarpinifolia','Populustremula' ,'Prunusavium','Quercuscerris', 'Quercusilex','Quercuspubescens']


    # nnome classe da trovare (ground truth)
    classedatrovare="Aceropalusobtusatum"
    #classedatrovare="castaneasativa"

    #classedatrovare="Castaneasativa"
    #classedatrovare="quercuspubescens"
    numeroclassecercata=classi_list.index(classedatrovare)
    print("num classe  cercata =" + str(numeroclassecercata))
    #destinazionesbagliata="/tmp/"+str(classedatrovare)+"-sbagliata/"
    #destinazionecorretta="/tmp/"+str(classedatrovare)+"-corretta/"
    #destinazionesbagliata="/home/lindo/Scaricati/gradcam-srgan/"+str(classedatrovare)+"-sbagliata/"
    #destinazionecorretta="/home/lindo/Scaricati/gradcam-srgan/"+str(classedatrovare)+"-corretta/"
    destinazionesbagliata="/home/lindo/Scaricati/gradcam-pievebovigliana-VIT/"+str(classedatrovare)+"-sbagliata/"
    destinazionecorretta="/home/lindo/Scaricati/gradcam-pievebovigliana-VIT/"+str(classedatrovare)+"-corretta/"

    descinazionecorretta= destinazionesbagliata="/tmp/jpg/gradcam/"

    dir_path="/tmp/jpg/"
     
    #dir_path="/home/lindo/programmi/castagno-univpm/dataset-google/DEFINITIVO/inaturalist-mixed-aerialview/test-srgan/"+str(classedatrovare)+"/"


    # ciclo per ogni file 
    """
    with open('csvmattonelle-VIT-0702_2025.csv', mode='w' , encoding='utf-8') as csvfile:
    # csvimage_writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE)
    # csvimage_writer = csv.writer(csvfile , delimiter=',')
     csvimage_writer = csv.writer(csvfile)
    #fields=['NameFile', 'Class-n1', 'Class-n2', 'Class-n3','Class-n4', 'Class-n5', 'Class-n6','Class-n7', 'Class-n8', 'Class-n9'  ] 
    #fields=[nomezona,'NameFile', 'Class-n1', 'Class-n2', 'Class-n3','Class-n4', 'Class-n5', 'Class-n6','Class-n7', 'Class-n8', 'Class-n9','vuoto','Perc-class1','Perc-class2','Perc-class3','Perc-class4','Perc-class5','Perc-class6','Perc-class7','Perc-class8','Perc-class9'] 
    #csvimage_writer.writerow(fields)
     fields=['NomeZona','NomeFileJPG','abb_ACeropalus', 'abb_CastaneaSativa', 'abb_Fraxinus', 'abb_OstryaCarpinifolia','abb_PopulusTremula', 'abb_PrunusAvium', 'abb_QuercusCerris','abb_Quercusilex', 'abb_Quercuspubescens',] 
     csvimage_writer.writerow(fields)
    """
    # genera listanomi tile in base a coordinate  y e x 
    #filex=input("Inserisci data e di mattonelle separati da - =")
    listafile=[]
    listaprimaclasse=[]
    xmax=12
    ymax=12
    #for x in range(xmax):
    # for  y in range(ymax):
    ppstrani=0  


    # for ognifile in  listafile:
    for ognifile in sorted(os.listdir(dir_path)):
      img_path=dir_path+ognifile
      print(str(ognifile))
      total_samples = total_samples+1


    #img_path = '/home/lindo/programmi/castagno-univpm/dataset-google/DEFINITIVO-3CLASSI/inaturalist-mixed-aerialview/test/castaneasativa/romano56.jpeg'
    #img_path = '/home/lindo/programmi/castagno-univpm/dataset-google/DEFINITIVO-3CLASSI/inaturalist-mixed-aerialview/test/oleaeuropeae/20230312_08394420.jpg'
      img_tensor = read_tensor(img_path)
      pp, cc = torch.topk(nn.Softmax(dim=1)(model(img_tensor.cuda())), numeroclassi)
    #  pp, cc = torch.max(model(img_tensor.cuda())  ,dim=1)

        #pp, cc = torch.topk(nn.ReLU()(model(img_tensor.cuda())), 4)
      max_class = cc[0][0].item()
      if max_class>10:
       ppstrani=ppstrani+1
    #  max_class = cc[0].item()
      print("CC = " ,cc)
      print( "PP =" ,pp)
      stringacsv=str(ognifile)
      stringaclassi=""
      listacsv=[stringacsv]
    # for i in range(cc[0].item()):
      for i in range(9):
        #stringaclassi=stringaclassi+","+str(cc[0][i].item())
        #listacsv.append(str(cc[0][i].item()))
        listacsv.append(str(labels[cc[0][i].item()]))

      listaprimaclasse.append(cc[0][0].item())
      listacsv.append("percentuali")
    # aggiungo percentuale attivazione 

      for i in range(9):
        scrivip=round(100*float(pp[0][i].item()))
        print("classe = " ,i, "percentuale=" ,scrivip)
        listacsv.append(str(scrivip))

      #stringacsv=stringacsv+stringaclassi
      print("scrivo stringa su file csv = ",stringacsv)
    #  csvimage_writer.writerow(stringacsv,)
      #csvimage_writer.writerow(listacsv)
      if (max_class < 10):
        print("classe predetta =" +str(max_class))

        if (max_class != numeroclassecercata):
            #copia file su cartella apposita
            destinazione=destinazionesbagliata

        else:
            total_correct=total_correct+1
            destinazione = destinazionecorretta

        plt.figure(figsize=(15, 5))
        for i, (p, c) in enumerate(zip(pp[0], cc[0])):
            plt.subplot(2, 5, i + 1)
            #sal = GradCAM(img_tensor, int(c), features_fn, classifier_fn)
            print(str(img_path))
            img = Image.open(img_path)

            #sal = Image.fromarray(sal)
            #sal = sal.resize(img.size, resample=Image.BILINEAR)

            #plt.title('{}: {:.1f}%'.format(get_class_name(c), 100 * float(p)))
            plt.title('{}: {:.1f}%'.format(c, 100 * float(p)))
            #plt.title('{}: {:.1f}%'.format("aceropalus","castaneasativa", "oleaeuropeae" , 100 * float(p)))
            plt.axis('off')
            stringatitolo= "Activation map - class:0=acero|1=castagno|2=fraxinus|3=carpinonero|4=pioppo|5=ciliegio|6=qcerris|7=qilex|8=roverella"+"\n"
            #plt.suptitle(stringatitolo)
            #plt.imshow(img)
            #plt.imshow(np.array(sal), alpha=0.5, cmap='jet')
            #plt.subplot(2,1,2)
            #img2 = mpimg.imread(img_path)
            #plt.imshow(img2)

        #plt.show()
        valorerandom=random.randint(20,140000)
        #nomefile =str(destinazione)+"GRADCAM-Pievebovigliana-mattonelle-"+str(ognifile)+"-predicted-"+ str(classi_list[max_class])+".png"
        #plt.savefig(nomefile)
        #plt.close()
        #plt.close(img2)


    #    destinazioneimmagine = str(destinazione) +"Pievebov-mattonelle-immaginetest-GroundTruth-"+str(classedatrovare)+"-predicted-" + str(classi_list[max_class]) + "-" +str(valorerandom) +".png"
    #    shutil.copyfile(img_path, destinazioneimmagine)

    # print data in CSV file
      #  nomefileimage=str(ognifile)
      #  csvimage_writer.writerow(nomefileimage)


    #    print("file analizzati = " +str(total_samples) )
    #accuracy = total_correct / total_samples
    #print( "accuracy = " + str(accuracy))
    fields=['NomeZona','NomeFileJPG','abb_ACeropalus', 'abb_CastaneaSativa', 'abb_Fraxinus', 'abb_OstryaCarpinifolia','abb_PopulusTremula','abb_PrunusAvium','abb_QuercusCerris','abb_Quercusilex', 'abb_Quercuspubescens',] 
    nomeinput=ognifile.split("_tile")[0]
    nomefileout=nomezona+"-abbondanzaVIT-dataset2024-size224NOpad-"+nomeinput+"-"+data+"-210225.txt"
    f = open(nomefileout, "w") 
    #scrivi nome immagine analizzata su file
    f.write(nomeinput)
    numelem=len(listaprimaclasse)
    print("num mattonelle =",numelem)
    listavitcsv=[]
    listavitcsv.append(nomezona)
    listavitcsv.append(nomefilejpgvit)
    nomefilecsvvit="abbondanza-dataset2024-size224noPAD-"+nomezona+"-VIT-"+data+".csv"
    with open(nomefilecsvvit, mode='a' , encoding='utf-8', newline='') as csvfile:
     csvimage_writer = csv.writer(csvfile)
     #csvimage_writer.writeheader()
     #csvimage_writer.write(fields)

     for m in range(9):
      res = sum(1 for i in  listaprimaclasse if i ==m)
    # print("classe " +str(labels[m])+  " num mattonelle primaclasse = " +str(res))
      perc=round(100*float(res/numelem),1)
      stringa="classe " +str(labels[m])+  " abbondanza = " +str(perc)
      print(stringa)
      stringa=stringa+"\n"
      f.write(stringa)
      listavitcsv.append(str(perc))
      #listavitcsv=listavitcsv+";"+str(perc)
      #listavitcsv.append()
     f.close()
      

  #fields=['NomeZona','NomeFileJPG','ACeropalus', 'CastaneaSativa', 'Fraxinus', 'OstryaCarpinifolia','PopulusTremula', 'PrunusAvium', 'QuercusCerris','Quercusilex', 'Quercuspubescens',] 
     csvimage_writer.writerow(listavitcsv)
     #print("Id classe conteggio  valori STRANI maggiori 9 =" , ppstrani)
     csvfile.close()


    #
# funzioni mattonelle versione 

# afggiungere padding 


from PIL import Image
import numpy as np
import math

def riduci50(img):
# riduce del 50% la dimensione immagine..usare per test 
#  riduci50="si"
  riduci50="no"
  if riduci50=="si":
   half = 0.5
   img2 = img.resize( [int(half * s) for s in img.size] , resample=Image.LANCZOS)
   print("RIDUCI IMG")
  return img2   





def add_pad(img):
#  aggiungipad="si"
  aggiungipad="no"
  print("AGGIUNGI FUNZIONE GESTIONE PADDING ")
  img = Image.open(img_nome)
  #img=riduci50(img)
  img_arr = np.array(img)  # Convert image to NumPy array

  # Calculate padding required to reach a multiple of tile_size
  # usare pil imageops.expand per aggiungere bord
# calcolare valore finale usando ceil per approsimare a valore intero successivo   nuovaalteza = tilesize*ceil (oldaltez/tilesize)

  newaltezza=tile_size* math.ceil(img_arr.shape[0]/ tile_size)
  newbase = tile_size* math.ceil(img_arr.shape[1]/ tile_size)
  print(" nuova base, nuova altezza = " , newbase, newaltezza)
   



#  pad_width_top, pad_width_bottom= ((newaltezza - img_arr.shape[0])//2)
  if aggiungipad=="si":
   pad_width_top = ((newaltezza - img_arr.shape[0])//2)
   pad_width_left = ((newbase - img_arr.shape[1])//2)
#  print(" PAD laterale  base, PAD   altezza = " , pad_width_left, pad_width_top)
  
#  pad_width = ((tile_size - (img_arr.shape[0] % tile_size)) % tile_size,
#               (tile_size - (img_arr.shape[1] % tile_size)) % tile_size)
#  pad_width_left, pad_width_right = pad_width[0] // 2, pad_width[0] - (pad_width[0] // 2)
#  pad_width_top, pad_width_bottom = pad_width[1] // 2, pad_width[1] - (pad_width[1] // 2)

  # Pad the image with zeros using constant_pad
   padded_img_arr = np.pad(img_arr, ((pad_width_top, pad_width_top), (pad_width_left, pad_width_left), (0, 0)), constant_values=0)
   padded_img = Image.fromarray(padded_img_arr)
  #resize image 
  new_image = img.resize((newbase, newaltezza))
  # Convert padded array back to PIL Image
  #padded_img = Image.fromarray(padded_img_arr)
  #print("dimensioni immagine nuova =" ,padded_img_arr.shape)
  if aggiungipad=="si":
    new_image = padded_img
  return new_image

def create_tiles(image_path, tile_size):
  """
  Creates 256x256 tiles from an image and saves them.

  Args:
      image_path: Path to the input image.
      tile_size: Size of each tile (default: 256).
  """

  # Open the image
  imgorig = Image.open(img_nome)
  width, height = imgorig.size
  print("dimensioni immagine originale =",imgorig.size)
  # 
  img= add_pad(imgorig)
  img.save("/tmp/imgpadded.jpg")
  width, height = img.size

  # Calculate the number of tiles in each dimension (considering edge cases)
  num_tiles_x = (width + tile_size - 1) // tile_size
  num_tiles_y = (height + tile_size - 1) // tile_size
  print(" x tile - y tile = ", num_tiles_x , num_tiles_y)
  # Loop through each tile and create a new image
  ntile=0
  for x in range(num_tiles_x):
    for y in range(num_tiles_y):
      ntile=ntile+1
      # Calculate the coordinates of the tile
      left = x * tile_size
      top = y * tile_size
      right = min(left + tile_size, width)
      bottom = min(top + tile_size, height)

      # Create a new image for the tile
      box = (x * tile_size, y * tile_size, (x + 1) * tile_size, (y + 1) * tile_size)
#      tile = img.crop((left, top, right, bottom))
      tile = img.crop(box)

      # Optionally resize the tile if needed (uncomment if desired)
      # resized_tile = tile.resize((tile_size, tile_size), Image.ANTIALIAS)
      
    # modifica x e y nel formato con lo 0 davanti
      

      # Save the tile with a descriptive filename

#      tile_filename = "/tmp/jpg/"+nomeinputfile+f"_tile_{x:02}_{y:02}.jpg"

#      tile_filename = "/tmp/jpg/"+nomeinputfile+"_tile_"+x+"_"+y+".jpg"

#      tile_filename = f"/tmp/jpg/DJI_0087_tile_{x:02}_{y:02}.jpg"
      nomeinputfile2=filenamejpg.split(".")[0]
      tile_filename = f"/tmp/jpg/{nomeinputfile2}_tile_{x:02}_{y:02}.jpg"

#      tile_filename = f"/tmp/jpg/IMG01_tile{ntile}_{x}_{y}.jpg"
      tile.save(tile_filename)


# MAIN PRINCIPALE
 # per ogni file nella cartella : crea mattonelle, eseguire test vgg19 e res101 , esegui VIT  


    #MAIN MATTONELLE

def main_mattonelle(fileinput):


    # Example usage
    #tile_size=128

    jpg_path="/tmp/jpg/"
    import os
    for filename in os.listdir(jpg_path):
        file_path = os.path.join(jpg_path, filename)
        os.remove(file_path)


    # xdim=input("TILE SIZE 224 oR 256 or 180? -->")
    xdim="224"
#    xdim="256"
#    tile_size=xdim
    if xdim=="256":
     print("size 256")
     tile_size=256
    if xdim=="224":
     print("size 224")
     tile_size=224
    if xdim=="180":
     print("size 180")
     tile_size=180



    create_tiles(fileinput,tile_size)


# MAIN 

import os
for filenamejpg in sorted(os.listdir(img_pathglobal)):
    # crea mattonelle, esegui vgg19 -res101 -VIT
    img_nome=img_pathglobal+filenamejpg
    main_mattonelle(img_nome)
    listamodello=["vgg19","vgg16","resnet101"]
    for mod in listamodello:
     main_vggresnet(mod,img_nome,filenamejpg)
    test_vit(filenamejpg)
    print("FINE")



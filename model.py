import pandas as pd
import numpy as np
import numpy.linalg as nl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pye57
import time
import laspy
import os


import tensorflow as tf 
import numpy as np
from tensorflow import keras
import ezdxf 
from ezdxf.addons import r12writer

model_path="final_model.h5"
model = keras.models.load_model(model_path)

model_path2="model_VGG.h5"
model2 = keras.models.load_model(model_path2)


#------------------------------------------TYPE 2 ---------------------------------------------------

def centrer13(d):
    X,Y,Z,c=d.cartesianX,d.cartesianY,d.cartesianZ,d.classe
    d13=d.loc[d.classe==13,'cartesianX':'classe']
    mx,my,mz=np.mean(d13.cartesianX),np.mean(d13.cartesianY),np.mean(d13.cartesianZ)
    del d13
    data,index,columns=np.array([X-mx,Y-my,Z-mz,c]).T,d.index,['cartesianX','cartesianY','cartesianZ','classe']
    del X,Y,Z,c
    dr=pd.DataFrame(data,index=index,columns=columns)
    return dr

def angle(XYi,XYf):
    x,y,xf,yf=XYi[0],XYi[1],XYf[0],XYf[1]
    Deno=x*x+y*y
    s,c=(yf*x-xf*y)/Deno,(xf*x+yf*y)/Deno
    angle=np.arccos(c)
    ss=np.arcsin(s)
    if ss>=0:
        return angle
    else :
        return -angle
    
def rotate(d,theta):
    XY=d.loc[:,'cartesianX':'cartesianY'].T
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    XYf,index,columns=np.dot(R,XY).T,d.index,['cartesianX','cartesianY']
    df=pd.DataFrame(XYf,index=index,columns=columns)
    df[['cartesianZ','classe']]=d.loc[:,'cartesianZ':'classe']
    return df


def files(d):
    pca=PCA(2)
    file1=d.loc[d.classe==13,'cartesianX':'cartesianY'].values
    file2=pca.fit_transform(file1)
    i=np.argmax(file2[:,0]).astype(int)
    XYi,XYf=file1[i,:],file2[i,:]
    return XYi,XYf

def preparer(d):
    d1=centrer13(d)
    a,b=files(d1)
    teta=angle(a,b)
    d2=rotate(d1,teta)
    return d2


def decouper(d,eps,alpha):
    d1=preparer(d)
    dx13=d1.loc[d1.classe==13,'cartesianX':'cartesianY']
    longx,mx,=max(dx13.cartesianX)-min(dx13.cartesianX),min(dx13.cartesianX)
    maxy,miny=max(dx13.cartesianY),min(dx13.cartesianY)
    d1=d1.loc[(d1.cartesianY<maxy+alpha)&(d1.cartesianY>miny-alpha),'cartesianX':'classe'] 
    nx=int(longx/eps)
    troncons=[]
    for i in range(nx):
        if i!=nx-1:
            dint=d1.loc[(d1.cartesianX>=mx+i*eps)&(d1.cartesianX<mx+(i+1)*eps),'cartesianX':'classe']
            maxy,miny=max(dint.loc[dint.classe==13,'cartesianY']),min(dint.loc[dint.classe==13,'cartesianY'])
            minz=min(dint.loc[dint.classe==13,'cartesianZ'])
            dint=dint.loc[(dint.cartesianY<maxy+alpha)&(dint.cartesianY>miny-alpha)&(dint.cartesianZ>minz-2.5*alpha),'cartesianX':'classe'] 
            img=image_2(dint)
            troncons.append(img)
        else:
            dint=d1.loc[(d1.cartesianX>=mx+i*eps)&(d1.cartesianX<mx+(i+2)*eps),'cartesianX':'classe']
            maxy,miny=max(dint.loc[dint.classe==13,'cartesianY']),min(dint.loc[dint.classe==13,'cartesianY'])
            dint=dint.loc[(dint.cartesianY<maxy+alpha)&(dint.cartesianY>miny-alpha)&(dint.cartesianZ>minz-2.5*alpha),'cartesianX':'classe'] 
            img=image_2(dint)
            troncons.append(img)
    return troncons,d1

def type2_partie1(d):
    eps,alpha=1.5,3.5
    L,d1=decouper(d,eps,alpha)
    del d
    return L,d1

def type2_partie2(L):
    c=[]
    for image in L:
        s=predict_2(image)
        c.append(s)
    return c

def coup(d1,i,j, mx):
    eps=1.5
    dr=d1.loc[(d1.cartesianX>=mx+i*eps)&(d1.cartesianX<mx+j*eps),'cartesianX':'classe']
    return dr
    
def type2_partie3(d,c):
    coupes=[]
    dx13=d.loc[d.classe==13,'cartesianX':'cartesianY']
    mx = min(dx13.cartesianX)
    del dx13
    a=0
    i1,i2=0,0
    for i in range(len(c)):
        if (c[i]==0) and (a==0):
            continue
        elif c[i]>0 and a==0:
            a+=1
            i1=i
        elif c[i]>0 and a>0:
            a+=1
            i2=i+1
        elif c[i]==0 and a>0:
            if a!=1:
                coupes.append(coup(d,i1,i2, mx))
            else:
                coupes.append(coup(d,i1-0.75,i1+1.75, mx))
            a=0
    return coupes
            
            
        
    

def image_2(de):
    M=np.mean(de.cartesianY)
    bordy=12
    width=120
    length=50
    if len(de) == 0 :
        return np.zeros(width*length*3).reshape(length,width,3)
    else :
        maxy=M+bordy
        miny=M-bordy
        pasy=(maxy-miny)/width
        diimg=de.loc[(de.cartesianY<maxy)&(de.cartesianY>miny),'cartesianY':'cartesianZ']
        minz=min(diimg.cartesianZ)
        maxz=minz+10
        pasz=(maxz-minz)/length
        dimg=diimg.loc[(diimg.cartesianZ<maxz)&(diimg.cartesianZ>minz),'cartesianY':'cartesianZ']
        Y=((dimg.cartesianY-miny)/pasy).astype(int)
        Z=((dimg.cartesianZ-minz)/pasz).astype(int)
        Image=np.zeros(width*length*3).reshape(length,width,3)
        for i,j in zip(Z,Y):
            Image[length-i-1,j]=[1,1,1]
        return Image

def predict_2(img):
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    return np.argmax(predictions)

def type2(data):
    images,d=type2_partie1(data)
    classes=type2_partie2(images)
    del images
    coupes=type2_partie3(d,classes)
    return coupes


#------------------------------------------TYPE 1 e57 ---------------------------------------------------

def e57_partie1(d):
  X,Y,Z=d.cartesianX,d.cartesianY,d.cartesianZ.values.reshape((len(d.cartesianZ),1))
  del d
  X,Y,Z=X-np.mean(X),Y-np.mean(Y),Z-np.mean(Z)
  X,Y=X.values.reshape((len(X),1)),Y.values.reshape((len(X),1))
  XY=np.concatenate((X,Y),axis=1)
  del X,Y
  pca=PCA(2)
  M=pca.fit_transform(XY)
  del XY
  data=np.concatenate((M,Z),axis=1)
  del M,Z
  d=pd.DataFrame(data,columns=['cartesianX','cartesianY','cartesianZ'])   
  return d    

def image(de):
    M=np.mean(de.cartesianY)
    bordy=12
    width=120
    length=50
    if len(de) == 0 :
        return np.zeros(width*length*3).reshape(length,width,3)
    else :
        maxy=M+bordy
        miny=M-bordy
        pasy=(maxy-miny)/width
        diimg=de.loc[(de.cartesianY<maxy)&(de.cartesianY>miny),'cartesianY':'cartesianZ']
        minz=min(diimg.cartesianZ)
        maxz=minz+10
        pasz=(maxz-minz)/length
        dimg=diimg.loc[(diimg.cartesianZ<maxz)&(diimg.cartesianZ>minz),'cartesianY':'cartesianZ']
        Y=((dimg.cartesianY-miny)/pasy).astype(int)
        Z=((dimg.cartesianZ-minz)/pasz).astype(int)
        Image=np.zeros(width*length*3).reshape(length,width,3)
        for i,j in zip(Z,Y):
            Image[length-i-1,j]=[1,1,1]
        return Image

def e57_partie2(d,eps):
  alpha=10
  d1=d.loc[(d.cartesianY<alpha)&(d.cartesianY>-alpha),'cartesianX':'cartesianZ'] 
  del d
  longx,mx=max(d1.cartesianX)-min(d1.cartesianX),min(d1.cartesianX)
  nx=int((longx/eps))
  troncons=[]
  for i in range(nx):
    if i!=nx-1:
      dint=d1.loc[(d1.cartesianX>=mx+i*eps)&(d1.cartesianX<mx+(i+1)*eps),'cartesianX':'cartesianZ']
      img=image(dint)
      del dint
      troncons.append(img)
    else:
      dint=d1.loc[(d1.cartesianX>=mx+i*eps)&(d1.cartesianX<mx+(i+2)*eps),'cartesianX':'cartesianZ'] 
      img=image(dint)
      del dint
      troncons.append(img)
  return troncons,d1

def predict(img):
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0)
  predictions = model2.predict(img_array)
  return np.argmax(predictions)


def e57_partie3(L):
  c=[]
  for image in L:
    s=predict(image)
    c.append(s)
  return c

def coup_e57(d1,i,j,eps):
  mx=min(d1.cartesianX)
  dr=d1.loc[(d1.cartesianX>=mx+i*eps)&(d1.cartesianX<mx+j*eps),'cartesianX':'cartesianZ']
  return dr
  
def e57_partie4(d,c,eps):
  coupes=[]
  a=0
  i1,i2=0,0
  for i in range(len(c)):
    if (c[i]==0) and (a==0):
      continue
    if c[i]>0 and a==0:
      a+=1
      i1=i
    if c[i]>0 and a>0:
      a+=1
      i2=i+1
    if c[i]==0 and a>0:
      if a!=1:
        coupes.append(coup_e57(d,i1,i2,eps))
      else:
        coupes.append(coup_e57(d,i1-0.75,i1+1.75,eps))
      a=0
  return coupes

def e57(d):
  eps=1.5
  d=e57_partie1(d)
  L,d1=e57_partie2(d,eps)
  del d
  c=e57_partie3(L)
  del L
  coupes=e57_partie4(d1,c,eps)
  return coupes



#------------------------------------------TYPE 1 las---------------------------------------------------
def meshagez_las(d,eps):
    zp,ip=d.loc[:,'cartesianZ'].values,d.index.values
    my,ly=min(d.cartesianZ),max(d.cartesianZ)-min(d.cartesianZ)
    l=[[] for i in range(int(ly/eps))]
    for i in range(len(ip)):
        j=int((zp[i]-my)/eps)
        if j>=0 and j<int(ly/eps):
            l[j].append(ip[i])
    return l
        
              
def cumulz_las(d,eps):
    l=meshagez_las(d,eps)
    L=[]
    for i in l:
        L.append(len(i))
    return L


def retour_indice_las(i,eps,my): # retourne le voisinage du max
    a,b=(i-1)*eps+my,(i+2)*eps+my
    return a,b


def centrer13_las(d):
    X,Y,Z,c=d.cartesianX,d.cartesianY,d.cartesianZ,d.classe
    d13=d.loc[d.classe==13,'cartesianX':'classe']
    mx,my,mz=np.mean(d13.cartesianX),np.mean(d13.cartesianY),np.mean(d13.cartesianZ)
    del d13
    data,index,columns=np.array([X-mx,Y-my,Z-mz,c]).T,d.index,['cartesianX','cartesianY','cartesianZ','classe']
    del X,Y,Z,c
    dr=pd.DataFrame(data,index=index,columns=columns)
    return dr

def angle_las(XYi,XYf):
    x,y,xf,yf=XYi[0],XYi[1],XYf[0],XYf[1]
    Deno=x*x+y*y
    s,c=(yf*x-xf*y)/Deno,(xf*x+yf*y)/Deno
    angle=np.arccos(c)
    ss=np.arcsin(s)
    if ss>=0:
        return angle
    else :
        return -angle
    
def rotate_las(d,theta):
    XY=d.loc[:,'cartesianX':'cartesianY'].T
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    XYf,index,columns=np.dot(R,XY).T,d.index,['cartesianX','cartesianY']
    df=pd.DataFrame(XYf,index=index,columns=columns)
    df[['cartesianZ','classe']]=d.loc[:,'cartesianZ':'classe']
    return df


def files_las(d):
    pca=PCA(2)
    file1=d.loc[d.classe==13,'cartesianX':'cartesianY'].values
    file2=pca.fit_transform(file1)
    i=np.argmax(file2[:,0]).astype(int)
    XYi,XYf=file1[i,:],file2[i,:]
    return XYi,XYf

def preparer_las(d):
    d1=centrer13_las(d)
    a,b=files_las(d1)
    teta=angle_las(a,b)
    d2=rotate_las(d1,teta)
    return d2

def decouper_type1_las(d,eps,alpha):#alpha=7,eps=1.5
    d1=preparer_las(d)
    del d
    dx13=d1.loc[d1.classe==13,'cartesianX':'cartesianZ']
    longx,mx,=max(dx13.cartesianX)-min(dx13.cartesianX),min(dx13.cartesianX)
    minZ = min(dx13.cartesianZ)
    d1=d1.loc[(d1.cartesianY<alpha)&(d1.cartesianY>-alpha)&(d1.cartesianZ > minZ - 1),'cartesianX':'classe'] 
    nx=int(longx/eps)
    troncons=[]
    for i in range(nx):
        if i!=nx-1:
            dint=d1.loc[(d1.cartesianX>=mx+i*eps)&(d1.cartesianX<mx+(i+1)*eps),'cartesianX':'classe']
            img=image_las(dint)
            troncons.append(img)
        else:
            dint=d1.loc[(d1.cartesianX>=mx+i*eps)&(d1.cartesianX<mx+(i+2)*eps),'cartesianX':'classe'] 
            img=image_las(dint)
            troncons.append(img)
    return troncons,d1

def image_las(de):
    M=np.mean(de.cartesianY)
    bordy=12
    width=120
    length=50
    if len(de) == 0 :
        return np.zeros(width*length*3).reshape(length,width,3)
    else :
        maxy=M+bordy
        miny=M-bordy
        pasy=(maxy-miny)/width
        diimg=de.loc[(de.cartesianY<maxy)&(de.cartesianY>miny),'cartesianY':'cartesianZ']
        minz=min(diimg.cartesianZ)
        maxz=minz+10
        pasz=(maxz-minz)/length
        dimg=diimg.loc[(diimg.cartesianZ<maxz)&(diimg.cartesianZ>minz),'cartesianY':'cartesianZ']
        Y=((dimg.cartesianY-miny)/pasy).astype(int)
        Z=((dimg.cartesianZ-minz)/pasz).astype(int)
        Image=np.zeros(width*length*3).reshape(length,width,3)
        for i,j in zip(Z,Y):
            Image[length-i-1,j]=[1,1,1]
        return Image

def predict_las(img):
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    return np.argmax(predictions)

def type2_partie2_las(L):
    c=[]
    for image in L:
        s=predict_las(image)
        c.append(s)
    return c

def coup_las(d1,i,j):
    eps=1.5
    dx13=d1.loc[d1.classe==13,'cartesianX':'cartesianY']
    mx=min(dx13.cartesianX)
    del dx13
    dr=d1.loc[(d1.cartesianX>=mx+i*eps)&(d1.cartesianX<mx+j*eps),'cartesianX':'classe']
    return dr
    
def type2_partie3_las(d,c):
    coupes=[]
    a = 0
    i1,i2=0,0
    for i in range(len(c)):
        if (c[i]==0) and (a==0):
            continue
        elif c[i]>0 and a==0:
            a+=1
            i1=i
        elif c[i]>0 and a>0:
            a+=1
            i2=i+1
        elif c[i]==0 and a>0:
            if a!=1:
                coupes.append(coup_las(d,i1,i2))
            else:
                coupes.append(coup_las(d,i1-0.75,i1+1.75))
            a=0
    return coupes

def type1_partie1_las(d):
    eps=0.1
    mz=min(d.cartesianZ)
    cz=cumulz_las(d,eps)
    j=np.argmax(np.array(cz))
    a,b=retour_indice_las(j,eps,mz)
    irail=d.loc[(d.cartesianZ>=a)&(d.cartesianZ<=b),'cartesianX':'cartesianZ'].index
    d['classe']=0
    d.loc[irail,'classe']=13
    return d


def type1_partie2_las(d):
    eps,alpha=1.5,9.5
    L,d1=decouper_type1_las(d,eps,alpha)
    c=type2_partie2_las(L)
    del L
    coupes=type2_partie3_las(d1,c)
    return coupes

def type1_las(data):
    d=type1_partie1_las(data)
    coupes=type1_partie2_las(d)
    return coupes


#------------------------------------------model ---------------------------------------------------


def extraction(d, filename, output, k):

    output = str(output)

    output_path = output.replace('\\','/') + '/'
     
    
    """
    data.append(np.array(pictures))
    
    classes=[]
    for i in range(len(pictures)):
        try :classe=int(input('class'))
        except ValueError:classe=int(input('class'))
        plt.imshow(pictures[i])
        plt.show()
        
        classes.append(classe)
    labels.append(classes)"""
    if k == 0 :
        Coupes = e57(d)
    elif k == 1 :
        Coupes = type1_las(d)
    else :
        Coupes = type2(d)

    for i in range(len(Coupes)):
        x = Coupes[i].iloc[:,0]
        y = Coupes[i].iloc[:,1]
        z = Coupes[i].iloc[:,2]

        file_name = filename[:-4] + '_' + str(i+1)
        placing_points = np.vstack((x, y, z)).transpose()
        output_file = output_path + file_name + '.dxf'
        with r12writer(output_file) as dxf:
            for point in placing_points:
                dxf.add_point(point)

    return len(Coupes)
    
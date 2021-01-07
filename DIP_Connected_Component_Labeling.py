import numpy as np
from numpy import pi, exp, sqrt
import cv2
from matplotlib import pyplot as plt
from random import randint
import timeit

def main():

    img = cv2.imread('pollen.bmp',cv2.IMREAD_GRAYSCALE)
    #######################################################
    #np_where()
    #img = fast_Connected_component_labeling(img)
    #img1 = fast_Connected_component_labeling_flatten(img)
    #img1 = Color_allocation_np(img1)
    #cv2.imshow('d', img1)
    #cv2.waitKey(0)
    img2 = fast_Connected_component_labeling_flatten_true(img)
    img2 = Color_allocation_np(img2)
    cv2.imwrite('pollen_fastest.jpg',img2)
    cv2.imshow('e',img2)
    #######################################################
    cv2.waitKey(0)
    #cv2.imshow('d', img1)
    #cv2.imwrite('symbol_fccl.jpg',img)

    return

def four_Connected_component_labeling(img):
    start = timeit.default_timer()
    img = np.array(img,dtype = np.uint8)
    y, x = img.shape
    Labeled_img = np.zeros_like(img)
    Labeled_img = np.array(Labeled_img, dtype = 'int16')
    cnt = 1
    pair_list =[]
    pair_max = []
    for i in range(y):
        for j in range(x):
            with np.errstate(divide='ignore', invalid='ignore'):
                 if img[i][j] == 0:
                    Labeled_img[i][j] = 0
                 else:
                     if Labeled_img[i-1][j] == 0 and Labeled_img[i][j-1] == 0:
                        Labeled_img[i][j] = cnt
                        cnt += 1
                     elif Labeled_img[i-1][j] != 0 and Labeled_img[i][j-1] == 0:
                        Labeled_img[i][j] = Labeled_img[i-1][j]
                     elif Labeled_img[i][j-1] != 0 and Labeled_img[i-1][j] == 0:
                        Labeled_img[i][j] = Labeled_img[i][j-1]
                     elif Labeled_img[i][j - 1] != 0 and Labeled_img[i - 1][j] != 0 :#and Labeled_img[i][j-1] != Labeled_img[i-1][j]:
                        Labeled_img[i][j] = Labeled_img[i-1][j]
                        pair_list.append((Labeled_img[i-1][j],Labeled_img[i][j-1]))
                        pair_max.append(Labeled_img[i][j-1])
                    #elif Labeled_img[i][j - 1] != 0 and Labeled_img[i - 1][j] != 0 and Labeled_img[i][j-1] == Labeled_img[i-1][j]:
                     #  Labeled_img[i][j] = Labeled_img[i][j-1]

    pair_list = list(set(pair_list))
    pair_max = list(set(pair_max))

    disjoint = Disjointset_pc(max(pair_max)+1)

    for p in range(len(pair_list)):
        disjoint.union(pair_list[p][0],pair_list[p][1])

    Labeled_img = Labeled_img.flatten()
    disjoint.data = np.array(disjoint.data, dtype='int16')
    data_set = list(set(disjoint.data))


    for h in range(1, len(data_set)):
        disjoint_index = np.where(disjoint.data == data_set[h])
        disjoint_index = np.asarray(disjoint_index)
        for t in range(disjoint_index.shape[1]):
            Labeled_img[Labeled_img == disjoint_index[0][t]] = data_set[h]



    Labeled_img=Labeled_img.reshape(y,x)

    stop = timeit.default_timer()
    print(stop - start)
#    for i in range(y):
 #       for j in range(x):
  #          for k in range(len(disjoint.data)):
   ##                Labeled_img[i][j] = disjoint.data[k]

    return Labeled_img

def Eight_Connected_component_labeling(img):

    img = np.array(img,dtype = np.uint8)

    y, x = img.shape

    Labeled_img = np.zeros((y,x+1),dtype = 'int16')

    cnt = 1
    pair_list =[]
    pair_max = []

    start = timeit.default_timer()
    for i in range(y):
        for j in range(x):
            with np.errstate(divide='ignore', invalid='ignore'):
                 if img[i][j] == 0:
                    Labeled_img[i][j] = 0
                 else:
                     if (Labeled_img[i-1][j],Labeled_img[i-1][j-1],Labeled_img[i-1][j+1],Labeled_img[i][j-1])==(0,0,0,0) :
                        Labeled_img[i][j] = cnt
                        cnt += 1

                    #case1
                     elif Labeled_img[i-1][j-1] != 0 and (Labeled_img[i][j-1],Labeled_img[i-1][j],Labeled_img[i-1][j+1]) == (0,0,0) :
                          Labeled_img[i][j] = Labeled_img[i-1][j-1]
                     elif Labeled_img[i-1][j] != 0 and (Labeled_img[i-1][j-1],Labeled_img[i-1][j+1],Labeled_img[i][j-1]) == (0,0,0):
                          Labeled_img[i][j] = Labeled_img[i-1][j]
                     elif Labeled_img[i-1][j+1] != 0 and (Labeled_img[i-1][j-1],Labeled_img[i][j-1],Labeled_img[i-1][j]) == (0,0,0):
                          Labeled_img[i][j] = Labeled_img[i-1][j+1]
                     elif Labeled_img[i][j-1] != 0 and (Labeled_img[i-1][j-1],Labeled_img[i-1][j],Labeled_img[i-1][j+1]) == (0,0,0):
                          Labeled_img[i][j] = Labeled_img[i][j-1]

                    #case2
                     elif Labeled_img[i-1][j-1] != 0 and Labeled_img[i-1][j+1] != 0 and Labeled_img[i-1][j] == 0 :
                          if Labeled_img[i][j-1] == 0:
                              Labeled_img[i][j] = Labeled_img[i-1][j-1]
                              pair_list.append((min(Labeled_img[i - 1][j - 1], Labeled_img[i - 1][j]),max(Labeled_img[i - 1][j - 1], Labeled_img[i - 1][j])))
                              pair_max.append(max(Labeled_img[i - 1][j - 1], Labeled_img[i - 1][j]))
                          elif Labeled_img[i][j-1] != 0:
                              Labeled_img[i][j] = Labeled_img[i][j-1]
                              pair_list.append((min(Labeled_img[i][j-1], Labeled_img[i-1][j+1]),max(Labeled_img[i][j-1], Labeled_img[i-1][j+1])))
                              pair_max.append(max(Labeled_img[i][j-1], Labeled_img[i-1][j+1]))

                     elif Labeled_img[i-1][j-1] != 0 and Labeled_img[i-1][j] != 0 and Labeled_img[i-1][j+1] == 0 :
                          if Labeled_img[i][j-1] == 0:
                              Labeled_img[i][j] = Labeled_img[i-1][j]
                          elif Labeled_img[i][j-1] != 0:
                              Labeled_img[i][j] = Labeled_img[i][j-1]

                     elif Labeled_img[i-1][j] !=0 and Labeled_img[i-1][j+1] !=0 and Labeled_img[i-1][j-1] == 0 :
                          if Labeled_img[i][j-1] == 0:
                              Labeled_img[i][j] = Labeled_img[i - 1][j + 1]
                          elif Labeled_img[i][j-1] !=0 :
                              Labeled_img[i][j] = Labeled_img[i][j - 1]
                              pair_list.append((min(Labeled_img[i][j-1], Labeled_img[i-1][j+1]),max(Labeled_img[i][j-1], Labeled_img[i-1][j+1])))
                              pair_max.append(max(Labeled_img[i][j-1],Labeled_img[i-1][j+1]))

                     elif Labeled_img[i-1][j-1] !=0 and Labeled_img[i][j-1] !=0 and (Labeled_img[i-1][j],Labeled_img[i-1][j+1])==(0,0):
                              Labeled_img[i][j] = Labeled_img[i][j-1]
                              pair_list.append((min(Labeled_img[i-1][j-1], Labeled_img[i][j-1]),max(Labeled_img[i][j-1], Labeled_img[i-1][j+1])))
                              pair_max.append(max(Labeled_img[i-1][j - 1], Labeled_img[i][j-1]))
                     elif Labeled_img[i-1][j] !=0 and Labeled_img[i][j-1] !=0  and (Labeled_img[i-1][j-1],Labeled_img[i-1][j+1])==(0,0):
                              Labeled_img[i][j] = Labeled_img[i][j-1]
                              pair_list.append((min(Labeled_img[i-1][j], Labeled_img[i][j-1]),max(Labeled_img[i-1][j], Labeled_img[i][j-1])))
                              pair_max.append(max(Labeled_img[i-1][j], Labeled_img[i][j-1]))
                     elif Labeled_img[i-1][j+1] !=0 and Labeled_img[i][j-1] !=0 and (Labeled_img[i-1][j],Labeled_img[i-1][j-1])==(0,0):
                            Labeled_img[i][j] = Labeled_img[i][j-1]
                            pair_list.append((min(Labeled_img[i-1][j+1],Labeled_img[i][j-1]),max(Labeled_img[i-1][j+1],Labeled_img[i][j-1])))
                            pair_max.append(max(Labeled_img[i-1][j+1], Labeled_img[i][j-1]))

                     elif Labeled_img[i-1][j-1] != 0 and Labeled_img[i-1][j] != 0 and Labeled_img[i-1][j+1] !=0 :
                          if Labeled_img[i][j-1] == 0 :
                             Labeled_img[i][j] = Labeled_img[i-1][j+1]
                          elif Labeled_img[i][j-1] != 0 :
                             Labeled_img[i][j] = Labeled_img[i][j-1]
                             pair_list.append((min(Labeled_img[i - 1][j + 1], Labeled_img[i][j - 1]),max(Labeled_img[i-1][j+1],Labeled_img[i][j-1])))
                             pair_max.append(max(Labeled_img[i - 1][j + 1], Labeled_img[i][j - 1]))
    stop = timeit.default_timer()
    print(stop - start)


    pair_list = list(set(pair_list))
    pair_max = list(set(pair_max))
    disjoint = Disjointset(max(pair_max)+1)

    for k in range(len(pair_list)):
        disjoint.union(pair_list[k][0],pair_list[k][1])

    #disjoint = Disjointset(max(pair_max)+1)

    #for k in range(len(pair_list)):
        #disjoint.union(pair_list[k][0],pair_list[k][1])

    for i in range(y):
        for j in range(x):
            for k in range(1,len(disjoint.data)):
                if Labeled_img[i][j] == k and disjoint.data[k] != k :
                   Labeled_img[i][j] = disjoint.data[k]


    return Labeled_img

def fast_Connected_component_labeling(img) :


#########################################################
    img = np.array(img, dtype='int16')
    y, x = img.shape
    Labeled_img = np.zeros((y, x + 1), dtype='int16')
    cnt = 1
    pair_list = []
    pair_list = np.array(pair_list,dtype ='int16')
########################################################## 0.0001
##########################################################
    start = timeit.default_timer()
    for i in range(y):
        for j in range(x):
                if img[i][j] != 0:
                    if Labeled_img[i-1][j] !=0 :
                        Labeled_img[i][j] = Labeled_img[i-1][j]
                    elif Labeled_img[i][j-1] !=0 :
                        Labeled_img[i][j] = Labeled_img[i][j-1]
                        if Labeled_img[i-1][j+1] != 0 :
                            pair_list = np.append(pair_list,[Labeled_img[i-1][j+1], Labeled_img[i][j-1]])
                    elif Labeled_img[i-1][j-1] != 0 :
                        Labeled_img[i][j] = Labeled_img[i-1][j-1]
                        if Labeled_img[i - 1][j + 1] != 0:
                            pair_list = np.append(pair_list,[Labeled_img[i - 1][j + 1], Labeled_img[i - 1][j - 1]])
                    #elif Labeled_img[i-1][j+1] !=0 and (Labeled_img[i-1][j-1], Labeled_img[i-1][j], Labeled_img[i][j-1])==(0,0,0) :
                    elif Labeled_img[i - 1][j + 1] != 0 and Labeled_img[i - 1][j - 1] ==0 and Labeled_img[i - 1][j] == 0 and Labeled_img[i][j - 1] == 0:
                        Labeled_img[i][j] = Labeled_img[i-1][j+1]
                    else :
                        Labeled_img[i][j] = cnt
                        cnt += 1
    stop = timeit.default_timer()
    print(stop - start)
######################################################### 0.30

    pair_max = np.max(pair_list)
    pair_list = pair_list.reshape((int(pair_list.shape[0]/2),2))
    disjoint = Disjointset_pc(pair_max + 1)

    for p in range(len(pair_list)):
        disjoint.union(pair_list[p][0], pair_list[p][1])

    Labeled_img = Labeled_img.flatten()
    disjoint.data = np.array(disjoint.data, dtype='int16')
    data_set = list(set(disjoint.data))

    for h in range(1, len(data_set)):
        disjoint_index = np.where(disjoint.data == data_set[h])
        disjoint_index = np.asarray(disjoint_index)
        for t in range(disjoint_index.shape[1]):
            Labeled_img[Labeled_img == disjoint_index[0][t]] = data_set[h]

    Labeled_img = Labeled_img.reshape(y, x+1)
    Labeled_img = Labeled_img[:,:x]

#########################################################0.002


#    for i in range(y):
    #       for j in range(x):
    #          for k in range(len(disjoint.data)):
    ##                Labeled_img[i][j] = disjoint.data[k]

    return Labeled_img

def fast_Connected_component_labeling_flatten(img) :


#########################################################
    img = np.array(img, dtype='int16')
    y, x = img.shape
    img = img.flatten()
    Labeled_img = np.zeros_like(img)


    cnt = 1
    pair_list = []
    pair_list = np.array(pair_list,dtype ='int16')

########################################################## 0.0001
##########################################################
    start = timeit.default_timer()
    for i in range(Labeled_img.shape[0]):
                if img[i] != 0:
                    if Labeled_img[i-x] !=0 :
                        Labeled_img[i] = Labeled_img[i-x]
                    elif Labeled_img[i-1] !=0 :
                        Labeled_img[i] = Labeled_img[i-1]
                        if Labeled_img[i-x+1] != 0 and Labeled_img[i-x+1] != Labeled_img[i-1]:
                            pair_list = np.append(pair_list,[Labeled_img[i-x+1], Labeled_img[i-1]])
                    elif Labeled_img[i-x-1] != 0 :
                        Labeled_img[i] = Labeled_img[i-x-1]
                        if Labeled_img[i-x+1] != 0 and Labeled_img[i-x+1] != Labeled_img[i-x-1]:
                            pair_list = np.append(pair_list,[Labeled_img[i-x+1], Labeled_img[i-x-1]])
                    #elif Labeled_img[i-1][j+1] !=0 and (Labeled_img[i-1][j-1], Labeled_img[i-1][j], Labeled_img[i][j-1])==(0,0,0) :
                    elif Labeled_img[i-x+1] != 0 and Labeled_img[i-x-1] ==0 and Labeled_img[i-x] == 0 and Labeled_img[i-1] == 0:
                        Labeled_img[i] = Labeled_img[i-x+1]
                    else :
                        Labeled_img[i] = cnt
                        cnt += 1

    stop = timeit.default_timer()
    print(stop - start)
######################################################### 0.30

    pair_max = np.max(pair_list)
    pair_list = pair_list.reshape((int(pair_list.shape[0]/2),2))
    disjoint = Disjointset_pc(pair_max + 1)

    for p in range(len(pair_list)):
        disjoint.union(pair_list[p][0], pair_list[p][1])

###########################################################
    #Labeled_img = Labeled_img.flatten()

    disjoint.data = np.array(disjoint.data, dtype='int16')
    data_set = list(set(disjoint.data))



############################################################
    start = timeit.default_timer()
    for h in range(1, len(data_set)):
        disjoint_index = np.where(disjoint.data == data_set[h])
        disjoint_index = np.asarray(disjoint_index)
        for t in range(disjoint_index.shape[1]):
            Labeled_img[Labeled_img == disjoint_index[0][t]] = data_set[h]

    Labeled_img = Labeled_img.reshape(y, x)


    stop = timeit.default_timer()
    print(stop - start)

#########################################################


#    for i in range(y):
    #       for j in range(x):
    #          for k in range(len(disjoint.data)):
    ##                Labeled_img[i][j] = disjoint.data[k]

    return Labeled_img

def fast_Connected_component_labeling_flatten_true(img):

    #########################################################
    img = np.array(img, dtype='int16')
    y, x = img.shape
    img = img.flatten()
    Labeled_img = np.zeros_like(img)

    cnt = 1
    pair_list = []
    pair_list = np.array(pair_list, dtype='int16')
    img_true = np.where(img > 0)
    img_true = np.asarray(img_true)
    ########################################################## 0.001
    ##########################################################
    start = timeit.default_timer()
    for i in range(img_true.shape[1]):
            v = img_true[0][i]
            v1 = v-x
            if Labeled_img[v1] != 0:
                Labeled_img[v] = Labeled_img[v1]
            elif Labeled_img[v - 1] != 0:
                Labeled_img[v] = Labeled_img[v - 1]
                if Labeled_img[v1+1] != 0 :# Labeled_img[v1+1] != Labeled_img[v - 1]:
                    pair_list = np.append(pair_list, [Labeled_img[v1+1], Labeled_img[v - 1]])
            elif Labeled_img[v1-1] != 0:
                Labeled_img[v] = Labeled_img[v1- 1]
                if Labeled_img[v1 + 1] != 0 :#and Labeled_img[v1 + 1] != Labeled_img[v1 - 1]:
                    pair_list = np.append(pair_list, [Labeled_img[v1 + 1], Labeled_img[v1 - 1]])
            # elif Labeled_img[i-1][j+1] !=0 and (Labeled_img[i-1][j-1], Labeled_img[i-1][j], Labeled_img[i][j-1])==(0,0,0) :
            elif Labeled_img[v1 + 1] != 0 and Labeled_img[v1 - 1] == 0 and Labeled_img[v1] == 0 and Labeled_img[v - 1] == 0:
                Labeled_img[v] = Labeled_img[v1 + 1]
            else :
                Labeled_img[v] = cnt
                cnt += 1

    stop = timeit.default_timer()
    print(stop - start)

    ######################################################### 0.30
    start = timeit.default_timer()
    disjoint = Disjointset_pc(np.max(pair_list) + 1)

    for p in range(0,pair_list.shape[0],2):
        disjoint.union(pair_list[p], pair_list[p+1])

    stop = timeit.default_timer()
    print(stop - start)
    ##########################################################
    start = timeit.default_timer()
    disjoint.data = np.array(disjoint.data, dtype='int16')
    disjoint_linear = np.arange(disjoint.data.shape[0])
    a = np.where(disjoint.data != disjoint_linear)
    a = np.asarray(a)
    for t in range(a.shape[1]) :
        c = a[0][t]
        b = np.where(Labeled_img==c)
        b = np.asarray(b)
        Labeled_img[b] = disjoint.data[c]

    #for h in range(1,data_set.shape[0]):
    #    disjoint_index = np.where(disjoint.data == data_set[h])
    #    disjoint_index = np.asarray(disjoint_index)
    #    data_set_h = data_set[h]
    #    for t in range (disjoint_index.shape[1]):
    #        Labeled_img[Labeled_img == disjoint_index[0][t]] = data_set_h

    Labeled_img = Labeled_img.reshape(y, x)
    stop = timeit.default_timer()
    print(stop - start)
    #########################################################

    #    for i in range(y):
    #       for j in range(x):
    #          for k in range(len(disjoint.data)):
    ##                Labeled_img[i][j] = disjoint.data[k]

    return Labeled_img

def Color_allocation_np(img):
    start = timeit.default_timer()
    img = np.array(img, dtype = 'int16')
    y, x = img.shape
    img1 = img
    img = img.flatten()
    val = np.unique(img)
    val = np.sort(val)
    mod = np.remainder(np.arange(val.shape[0]),5)

    dim0 = np.zeros((y,x)).flatten()
    dim1 = np.zeros((y,x)).flatten()
    dim2 = np.zeros((y,x)).flatten()

    for k in range(1,val.shape[0]):
        mod_k = mod[k]
        val_k = val[k]
        img_index = np.where(img == val_k)
        img_index = np.asarray(img_index)
        #print(val_k, img_index.shape[1])
        if mod_k == 1:
            dim0[img_index] = 255
        elif mod_k == 2:
            dim1[img_index] = 255
        elif mod_k == 3:
            dim2[img_index] = 255
        elif mod_k == 4:
            dim0[img_index] = 128
            dim1[img_index] = 128
        elif mod_k == 0:
            dim1[img_index] = 128
            dim2[img_index] = 128

    dim0 = dim0.reshape(y, x)
    dim1 = dim1.reshape(y, x)
    dim2 = dim2.reshape(y, x)
    BGR = np.stack((dim0, dim1, dim2), axis=2)
    stop = timeit.default_timer()
    print(stop - start)
    return BGR

def Color_allocation_np10(img):
    start = timeit.default_timer()
    img = np.array(img, dtype='int16')
    y, x = img.shape

    img = img.flatten()
    val = np.unique(img)
    val = np.sort(val)
    mod = np.remainder(np.arange(val.shape[0]), 10)

    dim0 = np.zeros((y, x)).flatten()
    dim1 = np.zeros((y, x)).flatten()
    dim2 = np.zeros((y, x)).flatten()

    for k in range(1, val.shape[0]):
        mod_k = mod[k]
        val_k = val[k]
        img_index = np.where(img == val_k)
        img_index = np.asarray(img_index)
        if mod_k == 1:
            dim2[img_index] = 255
            #print(mod_k, img_index.shape)
        elif mod_k == 2:
            dim2[img_index] = 255
            dim1[img_index] = 94
            #print(mod_k, img_index.shape)
        elif mod_k == 3:
            dim2[img_index] = 255
            dim1[img_index] = 228
            #print(mod_k, img_index.shape)
        elif mod_k == 4:
            dim2[img_index] = 171
            dim1[img_index] = 242
            #print(mod_k, img_index.shape)
        elif mod_k == 5:
            dim0[img_index] = 255
            dim1[img_index] = 216
            #print(mod_k, img_index.shape)
        elif mod_k == 6:
            dim0[img_index] = 255
            dim2[img_index] = 1
            #print(mod_k, img_index.shape)
        elif mod_k == 7:
            dim0[img_index] = 255
            dim2[img_index] = 95
            #print(mod_k, img_index.shape)
        elif mod_k == 8:
            dim0[img_index] = 221
            dim2[img_index] = 255
            #print(mod_k, img_index.shape)
        elif mod_k == 9:
            dim0[img_index] = 127
            dim2[img_index] = 255

            #print(mod_k, img_index.shape)
        elif mod_k == 0:
            dim0[img_index] = 88
            dim2[img_index] = 102

            #print(mod_k, img_index.shape)


    dim0 = dim0.reshape(y, x)
    dim1 = dim1.reshape(y, x)
    dim2 = dim2.reshape(y, x)
    BGR = np.stack((dim0, dim1, dim2), axis=2)
    stop = timeit.default_timer()
    print(stop - start)
    return BGR

def Color_allocation(img):
    img = np.array(img, dtype='int16')
    val = np.unique(img)
    val = np.sort(val)

    y, x = img.shape
    dim0 = np.zeros((y, x))

    BGR = np.stack((dim0, dim0, dim0), axis=2)

    for i in range(y):
        for j in range(x):
            for k in range(1, val.shape[0]):
                if img[i][j] == val[k]:
                    if k % 5 == 1:
                        BGR[i][j][0] = 255
                    elif k % 5 == 2:
                        BGR[i][j][1] = 255
                    elif k % 5 == 3:
                        BGR[i][j][2] = 255
                    elif k % 5 == 4:
                        BGR[i][j][0] = 128
                        BGR[i][j][1] = 128
                    elif k % 5 == 0:
                        BGR[i][j][1] = 128
                        BGR[i][j][2] = 128

    return BGR

class Disjointset_pc:

    def __init__(self,n):
        self.data  = np.arange(n)
        self.size  = n

    def upward(self,change_list,index):
        value = self.data[index]

        if value == index :
                return index

        change_list.append(index)
        return self.upward(change_list,value)

    def find(self,index):
        change_list =[]
        result = self.upward(change_list,index)

        for i in change_list :
            self.data[i] = result

        return result

    def union(self,x,y):

        x,y = self.find(x),self.find(y)

        if x == y :
            return

        self.data[self.data == y] = x

class Disjointset:

    def __init__(self,n):
        self.data  = np.arange(n)
        self.size  = n

    def find(self,index):

        return self.data[index]

    def union(self,x,y):

        x,y = self.find(x),self.find(y)

        if x == y :
            return

        self.data[self.data == y] = x

def np_where():
    a=[1,2,3,4]
    a= np.array(a,dtype= np.uint8)
    a= np.where(a>2,2,-2)

    print(a)
    return
main()




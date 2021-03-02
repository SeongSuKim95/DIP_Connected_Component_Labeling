
# This repository is python code of connected component labeling 

* Union and find Algorithm
    ```python
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

    ```
* Connected component labeling
    * Four connected component labeling(4-CCL)
      
      * Original image
      ![original](https://user-images.githubusercontent.com/62092317/106533707-01eed200-6536-11eb-9791-48681df419d2.PNG)
      * Low pass filter
      ![low_box](https://user-images.githubusercontent.com/62092317/106533689-fdc2b480-6535-11eb-9bfd-5ed3490adc85.PNG)
      ```python
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


        return Labeled_img
       ```   
   * Eight connected component labeling(8-CCL)
      * Before Normalizing
      ![Laplacian](https://user-images.githubusercontent.com/62092317/106533685-fd2a1e00-6535-11eb-926c-bd97658ffbbe.PNG)

      * After Nomalizing

      * Sharpening with normal laplacian filter
      ![Sharpening1](https://user-images.githubusercontent.com/62092317/106533713-02876880-6536-11eb-9ad1-77bbd3a69897.PNG)

      * Sharpening with diagonal laplacian filter
      ![Sharpening2](https://user-images.githubusercontent.com/62092317/106533716-031fff00-6536-11eb-854c-b817cba66683.PNG)

      * Comparison between normal & diagonal laplacian filter
      ![Sharpening](https://user-images.githubusercontent.com/62092317/106533709-01eed200-6536-11eb-9e4f-53dca2d3b200.PNG)
      ```python
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


        for i in range(y):
            for j in range(x):
                for k in range(1,len(disjoint.data)):
                    if Labeled_img[i][j] == k and disjoint.data[k] != k :
                    Labeled_img[i][j] = disjoint.data[k]


        return Labeled_img
      ```
    * Fast connected component labeling
      ![Sobel](https://user-images.githubusercontent.com/62092317/106533719-03b89580-6536-11eb-96e5-c9ccb0a80ffb.PNG)
      ![Sobel_vertical_horizontal](https://user-images.githubusercontent.com/62092317/106533722-04512c00-6536-11eb-9b3b-a70fab8fe350.PNG)
      ```python
            img = np.array(img, dtype='int16')
        y, x = img.shape
        Labeled_img = np.zeros((y, x + 1), dtype='int16')
        cnt = 1
        pair_list = []
        pair_list = np.array(pair_list,dtype ='int16')

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


        return Labeled_img
      ```
    
#See details in [HERE](https://github.com/SeongSuKim95/DIP_Various_Filter/blob/master/Spatial_and_Frequency_filter.pdf)


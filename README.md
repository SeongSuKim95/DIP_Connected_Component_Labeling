## Connected component labeling
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
      * Original image & Result
        ![4_connected_component_labeling1](https://user-images.githubusercontent.com/62092317/109629530-41d1c500-7b87-11eb-9a00-c59f13918a34.PNG)
        ![4_connected_component_labeling2](https://user-images.githubusercontent.com/62092317/109629538-4302f200-7b87-11eb-858f-8e7a4fd5b166.PNG)
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
      * Original image & Result
        ![8_connected_component_labeling](https://user-images.githubusercontent.com/62092317/109629540-439b8880-7b87-11eb-8291-460919a8b42d.PNG)
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
      * Algorithm
      ![Fast ccl_1](https://user-images.githubusercontent.com/62092317/109741791-24493d80-7c11-11eb-9e8d-f25dad03ede3.PNG)
      ![Fast ccl_2](https://user-images.githubusercontent.com/62092317/109741798-257a6a80-7c11-11eb-9201-2e6bb9b70780.PNG)
      ![Fast ccl_3](https://user-images.githubusercontent.com/62092317/109741800-257a6a80-7c11-11eb-8020-011039295e2a.PNG)
      * Code
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
   * Optimization result
     ![Optimization](https://user-images.githubusercontent.com/62092317/109629543-439b8880-7b87-11eb-8c98-e0f879754449.PNG)
     ![Pollen_Optimization](https://user-images.githubusercontent.com/62092317/109629546-44341f00-7b87-11eb-810b-e5db0e0c2cca.PNG)
     ![Result_for_otherimage](https://user-images.githubusercontent.com/62092317/109629552-44341f00-7b87-11eb-8ec1-c446b7022571.PNG)


#See details in [HERE](https://github.com/SeongSuKim95/DIP_Connected_Component_Labeling/blob/master/Connected_component_labeling.pdf)


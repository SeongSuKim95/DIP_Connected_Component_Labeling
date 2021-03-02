# DIP_various_filter
## This repository is python code of "Digital Image Processing, Rafael C.Gonzalez, 3rd edition" Chapter 3 & 4

* Spatial domain filtering
  * 2D Convolution
     ```python
     def Spatial_convolution(img, filter):
         m, n = filter.shape
         if (m == n):
             y, x = img.shape
             #y = y - m + 1
             #x = x - m + 1
             zp = int((m - 1) / 2)
             result = np.zeros((y, x))
             image_zero_padding = np.zeros((y+n-1,x+m-1))
             for i in range(y):
                 for j in range(x):
                     image_zero_padding[i+zp][j+zp] = img[i][j]

             for i in range(y):
                 for j in range(x):
                     result[i][j] = np.sum(image_zero_padding[i:i+m,j:j+n] * filter)
                     if result[i][j] <0:
                       result[i][j] = 0
             result = np.array(result,dtype=np.uint8)

     return result
     ```
  * Spatial filtering 
    * Box filter (Smoothing linear filter) 
      * Original image
  
      ![original](https://user-images.githubusercontent.com/62092317/106533707-01eed200-6536-11eb-9791-48681df419d2.PNG)
      * Low pass filter
      ![low_box](https://user-images.githubusercontent.com/62092317/106533689-fdc2b480-6535-11eb-9bfd-5ed3490adc85.PNG)
      ```python
      def S_smoothing_linear(img,c):

         smoothing_filter = np.ones((c,c))*(1/c**2)
         result = Spatial_convolution(img,smoothing_filter).astype(np.uint8)

         return  result
      
      def S_weighted_average(img):

         weighted_filter = np.array([[1,2,1],[2,4,2],[1,2,1]])*(1/16)
         result = Spatial_convolution(img,weighted_filter)
         result = np.array(result,dtype=np.uint8)
         
         return  result
     
      def S_Hpf(img,c):
      
         identity = np.zeros((c,c))
         center = int((c-1)/2)
         identity[center][center] = 1
         high_pass_filter = identity - np.ones((c,c))*(1/c**2)
         result = Spatial_convolution(img, high_pass_filter)

         result1 = np.array(result,dtype=np.uint8)
         result = logtransformation(result1)
         return result
     ```
   * Laplacian filter
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
      def S_Laplacian_filter(img):
          laplacian = np.array([[0,1,0],[1,-4,1],[0,1,0]])
          laplacian_diagonal = np.array([[1,1,1],[1,-8,1],[1,1,1]])
          result = Spatial_convolution(img,laplacian_diagonal)
          return result
      ```
    * Sobel operator
      ![Sobel](https://user-images.githubusercontent.com/62092317/106533719-03b89580-6536-11eb-96e5-c9ccb0a80ffb.PNG)
      ![Sobel_vertical_horizontal](https://user-images.githubusercontent.com/62092317/106533722-04512c00-6536-11eb-9b3b-a70fab8fe350.PNG)
      ```python
      def S_Sobel_horizontal(img):
          sobel_h = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
          result = Spatial_convolution(img,sobel_h)
          min = np.amin(result)
          result = result -min
          max = np.amax(result)
          result = (255 / max) * result
          result = np.array(result, dtype=np.uint8)
          
          return  result

      def S_Sobel_vertical(img):

         sobel_v = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
         result = Spatial_convolution(img,sobel_v)
         min = np.amin(result)
         result = result -min
         max = np.amax(result)
         result = (255 / max) * result
         result = np.array(result, dtype=np.uint8)
         
         return result

      ```
    
    * Unsharp masking and highboost filtering
      * Unsharp masking
      ![Unsharpmasking](https://user-images.githubusercontent.com/62092317/106533724-04e9c280-6536-11eb-8fba-da6128d47ed9.PNG)
      ```python
      def S_Unsharp_masking(img,c,a,k):

          a,b = img.shape
          #hpf_image = S_Hpf(img,c)
          lpf_image = S_Gaussian_LPF(img,c,a)
          mask_image = np.empty_like(img)
          mask_image = np.array(mask_image,dtype=float)
          img = np.array(img,dtype=float)
          lpf_image = np.array(lpf_image,dtype=float)
          for i in range(a):
              for j in range(b):
                  mask_image[i][j] = img[i][j] - lpf_image[i][j]
          result = np.empty_like(img)
          result = np.array(result, dtype=float)
          for i in range(a):
              for j in range(b):
                  result[i][j] = img[i][j] + k*mask_image[i][j]
          min = np.amin(result)
          result = result - min
          max = np.amax(result)
          result = (255 / max) * result
          result = np.array(result, dtype=np.uint8)

          return result
      ```
      * High boost by Unsharpmasking + Sobel filter
      ![Unsharpmasking+Sobel](https://user-images.githubusercontent.com/62092317/106533728-05825900-6536-11eb-9f22-4e46e9beeea5.PNG)

* Frequency domain filtering
  ```python
  def Fourier_transform(img):
    FT_transformed = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(FT_transformed)
    fft_shift = np.asarray(fft_shift)

    magnitude_spectrum = 20*np.log(np.abs(fft_shift))/np.log(5)
    magnitude_spectrum = np.asarray(magnitude_spectrum,dtype=np.uint8)

    return magnitude_spectrum
  
  ``` 
  * Ideal low&high-pass filter
    * Low pass filter
    ![Frequency_LPF](https://user-images.githubusercontent.com/62092317/106533676-fa2f2d80-6535-11eb-9a3d-cdd4fec60411.PNG)
    ```python
    def F_LPF_round(img,d):
    
        Lowpassfilter = np.zeros_like(img)
        a,b = Lowpassfilter.shape
        for x in range(0,b):
            for y in range(0,a):
                if int(sqrt((x-b/2)**2 + (y-a/2)**2)) <= d:
                    Lowpassfilter[y,x] = 1

        Lowpassfilter = np.array(Lowpassfilter,dtype= float)
        img = Fourier_transform(img)
        result = img * Lowpassfilter
        result1 = np.fft.ifft2(result)
        result2 = np.absolute(result1)

        max = np.amax(result2)
        result3 = (255 / max) * result2
        result4 = np.array(result3, dtype=np.uint8)
        # xx = np.linspace(0,img.shape[1]-1,img.shape[1])
        # yy = np.linspace(0,img.shape[0]-1,img.shape[0])
        #
        # xxx,yyy =np.meshgrid(xx,yy)
        # fig = plt.figure()
        # ax = plt.axes(projection ='3d')
        # ax.plot_surface(yyy,xxx,Lowpassfilter)
        # plt.show()

        return result4
        
    def F_LPF_square(img,c):

        Lowpassfilter = np.zeros_like(img)
        a,b = Lowpassfilter.shape
        if (a%2) == 0:
            if (b%2) == 0:
                Lowpassfilter[int(b/2)-c:int(b/2)+1+c,int(a/2)-c:int(a/2)+1+c] = 1
            elif (b%2) == 1:
                Lowpassfilter[int((b-1)/2)-c:int((b-1)/2)+2+c, int(a/2)-c:int(a/2)+1+c] = 1
        elif (a%2) == 1:
            if (b%2) == 0:
                Lowpassfilter[int(b/2)-c:int(b/2)+1+c,int((a-1)/2)-c:int((a-1)/2)+2+c]=1
            elif (b%2) ==1:
                Lowpassfilter[int((b-1)/2)-c:int((b-1)/2)+2+c,int((a-1)/2)-c:int((a-1)/2)+2+c] = 1

        Lowpassfilter = np.array(Lowpassfilter,dtype=float)
        img = Fourier_transform(img)
        result = img*Lowpassfilter
        result1 = np.fft.ifft2(result)
        result2 = np.absolute(result1)
        max = np.amax(result2)
        result3 = (255/max)*result2
        result4 = np.array(result3,dtype=np.uint8)

        return result4
    ```
    * High pass filter
    ![Frequency_HPF](https://user-images.githubusercontent.com/62092317/106533749-09ae7680-6536-11eb-8344-b3f9b46461f0.PNG)
    ```python
    def F_HPF_round(img,d):
    Highpassfilter = np.ones_like(img)
    a,b = Highpassfilter.shape
    for x in range(0,b):
        for y in range(0,a):
            if int(sqrt((x-b/2)**2 + (y-a/2)**2)) <= d:
                Highpassfilter[y,x] = 0

    Highpassfilter = np.array(Highpassfilter,dtype= float)
    img = Fourier_transform(img)
    result = img * Highpassfilter
    result1 = np.fft.ifft2(result)
    result2 = np.absolute(result1)
    max = np.amax(result2)
    result3 = (255 / max) * result2
    result4 = np.array(result3, dtype=np.uint8)

    return result4

    def F_HPF_square(img,c):
        Highpassfilter = np.ones_like(img)
        a, b = Highpassfilter.shape
        if (a % 2) == 0:
            if (b % 2) == 0:
                Highpassfilter[int(b / 2) - c:int(b / 2) + 1 + c, int(a / 2) - c:int(a / 2) + 1 + c] = 0
            elif (b % 2) == 1:
                Highpassfilter[int((b - 1) / 2) - c:int((b - 1) / 2) + 2 + c, int(a / 2) - c:int(a / 2) + 1 + c] = 0
        elif (a % 2) == 1:
            if (b % 2) == 0:
                Highpassfilter[int(b / 2) - c:int(b / 2) + 1 + c, int((a - 1) / 2) - c:int((a - 1) / 2) + 2 + c] = 0
            elif (b % 2) == 1:
                Highpassfilter[int((b - 1) / 2) - c:int((b - 1) / 2) + 2 + c, int((a - 1) / 2) - c:int((a - 1) / 2) + 2 + c] = 0

        Highpassfilter = np.array(Highpassfilter, dtype=float)
        img = Fourier_transform(img)
        result = img * Highpassfilter
        result1 = np.fft.ifft2(result)
        result2 = np.absolute(result1)
        max = np.amax(result2)
        result3 = (255 / max) * result2
        result4 = np.array(result3, dtype=np.uint8)
        return result4
    ```

  * Gaussian low&high-pass filter
    * Low pass filter
    ![Gaussian_LPF](https://user-images.githubusercontent.com/62092317/106533683-fbf8f100-6535-11eb-971b-92a339371d4a.PNG)
    ```python
    def F_Gaussian_LPF(img,sigma):

        a,b = img.shape

        Gaussian_height = [exp(-(z-int(a/2))*(z-int(a/2)) / (2 * sigma * sigma))  for z in range(0, a)]
        Gaussian_width = [exp(-(z-int(b/2))*(z-int(b/2)) / (2 * sigma * sigma))  for z in range(0, b)]

        Gaussian_filter = np.outer(Gaussian_height, Gaussian_width)
        Gaussian_filter = np.array(Gaussian_filter,dtype=np.float)/Gaussian_filter.sum(dtype=np.float)
        img = Fourier_transform(img)
        result = img*Gaussian_filter
        result1 = np.fft.ifft2(result)
        result2 = np.absolute(result1)
        max = np.amax(result2)
        result3 = (255/max)*result2
        result4 = np.array(result3,dtype=np.uint8)

        # print(img.shape)
        # xx = np.linspace(0,img.shape[1]-1,img.shape[1])
        # yy = np.linspace(0,img.shape[0]-1,img.shape[0])
        # 
        # xxx,yyy =np.meshgrid(xx,yy)
        # fig = plt.figure()
        # ax = plt.axes(projection ='3d')
        # ax.plot_surface(yyy,xxx,Gaussian_filter)

        #plt.show()
        return result4
    ```
    * High pass filter
    ![Gaussian_HPF](https://user-images.githubusercontent.com/62092317/106533680-fbf8f100-6535-11eb-96e5-33c2bdffba38.PNG)
    ```python
    def F_Gaussian_HPF(img,sigma):
        from numpy import pi, exp, sqrt
        a, b = img.shape

        Gaussian_height = [(exp(-(z-int(a/2))*(z-int(a/2))/(2*sigma*sigma))) for z in range(0, a)]
        Gaussian_width = [(exp(-(z-int(b/2))*(z-int(b/2))/(2*sigma*sigma))) for z in range(0, b)]

        Gaussian_filter = np.outer(Gaussian_height, Gaussian_width)
        Gaussian_filter = np.array(Gaussian_filter, dtype=np.float)
        Gaussian_filter = np.array(1-Gaussian_filter,dtype=np.float)
        img = Fourier_transform(img)
        result = img * (Gaussian_filter)
        result1 = np.fft.ifft2(result)
        result2 = np.absolute(result1)
        max = np.amax(result2)
        result3 = (255 / max) * result2
        result4 = np.array(result3, dtype=np.uint8)
        # print(img.shape)
        # 
        # xx = np.linspace(0,img.shape[1]-1,img.shape[1])
        # yy = np.linspace(0,img.shape[0]-1,img.shape[0])
        # 
        # xxx,yyy =np.meshgrid(xx,yy)
        # fig = plt.figure()
        # ax = plt.axes(projection ='3d')
        # ax.plot_surface(yyy,xxx,Gaussian_filter)
        # plt.show()

        return result4
    ```
  * Butterworth low&high-pass filter
   * Low pass filter
   ```python
   def Butterworth_LPF(img,d,order):

       Lowpassfilter = np.zeros_like(img)
       Lowpassfilter = np.array(Lowpassfilter,dtype=float)
       a,b = Lowpassfilter.shape
       for x in range(0,b):
           for y in range(0,a):
                   distance = sqrt((x-b/2)**2 + (y-a/2)**2)
                   Lowpassfilter[y,x] = 1/(1+(distance/d)**(2*order))

       img = Fourier_transform(img)
       result = img * Lowpassfilter
       result1 = np.fft.ifft2(result)
       result2 = np.absolute(result1)
       max = np.amax(result2)
       result3 = (255 / max) * result2
       result4 = np.array(result3, dtype=np.uint8)

       # xx = np.linspace(0,img.shape[1]-1,img.shape[1])
       # yy = np.linspace(0,img.shape[0]-1,img.shape[0])
       # 
       # xxx,yyy =np.meshgrid(xx,yy)
       # fig = plt.figure()
       # ax = plt.axes(projection ='3d')
       # ax.plot_surface(yyy,xxx,Lowpassfilter)
       # plt.show()
       return result4
   ```
   * High pass filter
   ![ButterWorth_HPF](https://user-images.githubusercontent.com/62092317/106533748-0915e000-6536-11eb-8e0b-2797eef4dbbf.PNG)
   ```python
   def Butterworth_HPF(img,d,order):
       Highpassfilter = np.zeros_like(img)
       Highpassfilter = np.array(Highpassfilter,dtype=float)
       a,b = Highpassfilter.shape
       for x in range(0,b):
           for y in range(0,a):
               distance = sqrt((x-b/2)**2+(y-a/2)**2)
               if distance==0 :
                   Highpassfilter[y,x] = 0
               else :
                   Highpassfilter[y,x] = 1/(1+(d/distance)**(2*order))

       img = Fourier_transform(img)
       result = img * Highpassfilter
       result1 = np.fft.ifft2(result)
       result2 = np.absolute(result1)
       max = np.amax(result2)
       result3 = (255 / max) * result2
       result4 = np.array(result3, dtype=np.uint8)
       # print(img.shape)
       # xx = np.linspace(0,img.shape[1]-1,img.shape[1])
       # yy = np.linspace(0,img.shape[0]-1,img.shape[0])
       # 
       # xxx,yyy =np.meshgrid(xx,yy)
       # fig = plt.figure()
       # ax = plt.axes(projection ='3d')
       # ax.plot_surface(yyy,xxx,Highpassfilter)
       # plt.show()

       return result4
   ```
  * Notch filter
    * Frequency components of original image & Required notch filter
    ![Notch_filter](https://user-images.githubusercontent.com/62092317/106533699-00250e80-6536-11eb-950e-bc9c26954890.PNG)
    * Filtered image
    ![Notch_filter_1](https://user-images.githubusercontent.com/62092317/106533701-00bda500-6536-11eb-9142-656d0226afca.PNG)
    * Noise of the image ( Extracted by Inverse notch filter)
    ![Notch_filter_2](https://user-images.githubusercontent.com/62092317/106533704-01563b80-6536-11eb-960a-8a72774c82ff.PNG)
    
    ```python
    def Notch_round_filter(img,d):
        a, b = img.shape
        Notchfilter = np.ones_like(img)

        for x in range(0,b):
            for y in range(0,a):
                if (sqrt((x-111)**2 + (y-81)**2)) <= d or (sqrt((x-55)**2 + (y-85)**2)) <= d or (sqrt((x-57)**2 + (y-165)**2)) <= d or (sqrt((x-113)**2 + (y-161)**2)) <= d or (sqrt((x-55)**2 + (y-44)**2)) <= d or (sqrt((x-111)**2 + (y-40)**2)) <= d or (sqrt((x-57)**2 + (y-206)**2)) <= d or(sqrt((x-113)**2 + (y-202)**2)) <= d :
                 Notchfilter[y,x] = 0

        Notchfilter = np.array(Notchfilter,dtype= float)
        img = Fourier_transform(img)
        result = img * Notchfilter
        result1 = np.fft.ifft2(result)
        result2 = np.absolute(result1)
        max = np.amax(result2)
        result3 = (255 / max) * result2
        result4 = np.array(result3, dtype=np.uint8)

        # xx = np.linspace(0,img.shape[1]-1,img.shape[1])
        # yy = np.linspace(0,img.shape[0]-1,img.shape[0])
        # 
        # xxx,yyy =np.meshgrid(xx,yy)
        # fig = plt.figure()
        # ax = plt.axes(projection ='3d')
        # ax.plot_surface(yyy,xxx,Notchfilter)
        # plt.show()

        return result4
    ```

* Restoration
  * Noise Function
  ```python
     def Bluring_Noise(img,k):

       Noisefilter = np.zeros_like(img)
       Noisefilter = np.array(Noisefilter, dtype=float)
       a, b = Noisefilter.shape

       for x in range(0, b):
           for y in range(0, a):
                   Noisefilter[y, x] = exp((-1)*k*((y-b/2)**2 +(x-a/2)**2)**(5/6))

       img = Fourier_transform(img)
       result = img * Noisefilter
       result1 = np.fft.ifft2(result)
       result2 = np.absolute(result1).astype(np.uint8)

       return result2
  ```
  * Spatial domain restoration
    * Median filter
    ![median_filter](https://user-images.githubusercontent.com/62092317/106533692-fe5b4b00-6535-11eb-8169-186ddca92125.PNG)
    ```python
    def Median_filter(img,c):

        zp = int((c-1)/2)
        y, x = img.shape
        image_zero_padding = np.zeros((y + c- 1, x + c - 1))
        for i in range(y):
            for j in range(x):
                image_zero_padding[i+zp][j+zp] = img[i][j]

        image_zero_padding = np.array(image_zero_padding, dtype=np.uint8)

        filter = np.zeros((c, c))
        result = np.zeros((y, x))

        for i in range(y):
            for j in range(x):
                filter = image_zero_padding[i :i+2*zp+1, j:j+2*zp+1]
                result[i][j] = np.median(filter)

        result = np.array(result, dtype=np.uint8)

    return result
    ```
    * Min-max filter
      * Min filter
      ![min_filter](https://user-images.githubusercontent.com/62092317/106533697-ff8c7800-6535-11eb-81a3-d202f0a5bc86.PNG)
      ```python
      def Min_filter(img,c):
          zp = int((c-1)/2)
          y, x = img.shape
          image_zero_padding = np.zeros((y + c- 1, x + c - 1))
          for i in range(y):
              for j in range(x):
                  image_zero_padding[i+zp][j+zp] = img[i][j]

          image_zero_padding = np.array(image_zero_padding, dtype=np.uint8)

          filter = np.zeros((c, c))
          result = np.zeros((y, x))

          for i in range(y):
              for j in range(x):
                  filter = image_zero_padding[i :i+2*zp+1, j:j+2*zp+1]
                  result[i][j] = np.amin(filter)

          result = np.array(result, dtype=np.uint8)

          return result
      ```
      * Max filter
      ![max_filter](https://user-images.githubusercontent.com/62092317/106533690-fdc2b480-6535-11eb-944b-4eefceba55c6.PNG)
      ```python
      def Max_filter(img,c):
          zp = int((c-1)/2)
          y, x = img.shape
          image_zero_padding = np.zeros((y + c- 1, x + c - 1))
          for i in range(y):
              for j in range(x):
                  image_zero_padding[i+zp][j+zp] = img[i][j]

          image_zero_padding = np.array(image_zero_padding, dtype=np.uint8)

          filter = np.zeros((c, c))
          result = np.zeros((y, x))

          for i in range(y):
              for j in range(x):
                  filter = image_zero_padding[i :i+2*zp+1, j:j+2*zp+1]
                  result[i][j] = np.amax(filter)

          result = np.array(result, dtype=np.uint8)

          return result
      ```
    * Midpoint filter
    ![mid_point_filter](https://user-images.githubusercontent.com/62092317/106533693-fef3e180-6535-11eb-8516-6b490d66e4a2.PNG)
    ```python
    def Midpoint_filter(img,c):
        zp = int((c - 1) / 2)
        y, x = img.shape
        image_zero_padding = np.zeros((y + c - 1, x + c - 1))
        for i in range(y):
            for j in range(x):
                image_zero_padding[i + zp][j + zp] = img[i][j]

        image_zero_padding = np.array(image_zero_padding, dtype=np.uint8)

        filter = np.zeros((c, c))
        result = np.zeros((y, x))

        for i in range(y):
            for j in range(x):
                filter = image_zero_padding[i:i + 2 * zp + 1, j:j + 2 * zp + 1]
                result[i][j] = (1/2)*np.amax(filter)+(1/2)*np.amin(filter)
                a= np.amax(filter)
                b= np.amin(filter)
        result = np.array(result, dtype=np.uint8)

        return result
    ```
    * Alpha-trimmed mean filter
    ![Alpha_trimmed_mean_filter](https://user-images.githubusercontent.com/62092317/106533743-087d4980-6536-11eb-8add-b9f638ae5ac7.PNG)
    ```python
    def Alpha_trimmed_mean_filter(img,c,a):

        zp = int((c - 1) / 2)
        y, x = img.shape
        image_zero_padding = np.zeros((y + c - 1, x + c - 1))
        for i in range(y):
            for j in range(x):
                image_zero_padding[i + zp][j + zp] = img[i][j]

        image_zero_padding = np.array(image_zero_padding, dtype=np.uint8)

        filter = np.zeros((c, c))
        result = np.zeros((y, x))

        for i in range(y):
            for j in range(x):
                filter = image_zero_padding[i:i + 2 * zp + 1, j:j + 2 * zp + 1]
                ordered_filter = np.array(filter).reshape(c**2,)
                ordered_filter = np.sort(ordered_filter)
                ordered_trimmed_filter =  ordered_filter[a:c**2-a]
                result[i][j]=(1/(c**2 - 2*a))*np.sum(ordered_trimmed_filter)

        result = np.array(result, dtype=np.uint8)

        return result
    ```
    * Adaptive median fitler
    ![Adaptive_median_filter](https://user-images.githubusercontent.com/62092317/106533738-074c1c80-6536-11eb-9fc9-58109823a3ca.PNG)
    ```python
    def Adaptive_median_filter(img):
        Smax = 7
        zpm = 3
        y, x = img.shape
        image_zero_padding = np.zeros((y + Smax - 1, x + Smax - 1))
        for i in range(y):
            for j in range(x):
                image_zero_padding[i+zpm][j+zpm] = img[i][j]

        image_zero_padding = np.array(image_zero_padding,dtype =np.uint8)

        filter = np.zeros((3,3))
        filter1 = np.zeros((5,5))
        filter2 = np.zeros((7,7))

        result = np.zeros((y, x))
        for i in range(y):
            for j in range(x):
                filter = image_zero_padding[i+2:i+5,j+2:j+5]
                if (np.median(filter) - np.amin(filter)) > 0 and (np.median(filter) - np.amax(filter)) < 0:
                    if image_zero_padding[i + 3][j + 3] - np.amin(filter) > 0 and image_zero_padding[i + 3][j + 3] - np.amax(filter) < 0:
                        result[i][j] = image_zero_padding[i + 3][j + 3]
                    else:
                        result[i][j] = np.median(filter)
                else:
                    filter1 =image_zero_padding[i+1:i+6,j+1:j+6]
                    if (np.median(filter1)-np.amin(filter1))>0 and (np.median(filter1)-np.amax(filter1)) < 0 :
                        if image_zero_padding[i+3][j+3] - np.amin(filter1) >0 and image_zero_padding[i+3][j+3] -np.amax(filter1) <0:
                            result[i][j] = image_zero_padding[i+3][j+3]
                        else :
                            result[i][j] = np.median(filter1)
                    else :
                        filter2 = image_zero_padding[i:i+7,j:j+7]
                        if (np.median(filter2) - np.amin(filter2)) > 0 and (np.median(filter2) - np.amax(filter2)) < 0:
                            if image_zero_padding[i + 3][j + 3] - np.amin(filter2) > 0 and image_zero_padding[i + 3][j + 3] - np.amax(filter2) < 0:
                                result[i][j] = image_zero_padding[i + 3][j + 3]
                            else:
                                result[i][j] = np.median(filter2)


        result = np.array(result,dtype=np.uint8)

        return result
    ```
  * Frequency domain restoration
    * Inverse filtering
    ![Inverse_filter](https://user-images.githubusercontent.com/62092317/106533684-fc918780-6535-11eb-83a5-842641c1fdee.PNG)
    ```python
    def Inverse_filter(img,k):
        Inverse_filter = np.zeros_like(img)
        Inverse_filter = np.array(Inverse_filter, dtype=float)
        a, b = Inverse_filter.shape

        for x in range(0, b):
            for y in range(0, a):
                Inverse_filter[y,x] =1 /(exp((-1)*k*((y-b/2)**2+(x-a/2)**2)**(5/6)))

        img = Fourier_transform(img)
        result = img * Inverse_filter
        result1 = np.fft.ifft2(result)
        result2 = np.absolute(result1)

        max = np.amax(result2)
        result3 = (255 / max ) * result2
        result4 = np.array(result3, dtype=np.uint8)

        return result4
        
    def Inverse_with_Butterworth_filter(img,k,r):
        Inverse_filter = np.zeros_like(img)
        Inverse_filter = np.array(Inverse_filter, dtype=float)
        a, b = Inverse_filter.shape

        for x in range(0, b):
            for y in range(0, a):
                Inverse_filter[y, x] =1 /(exp((-1)*k* ((y - b / 2) ** 2 + (x - a / 2) ** 2) ** (5 / 6)))

        Lowpassfilter = np.zeros_like(img)
        Lowpassfilter = np.array(Lowpassfilter,dtype=float)
        c,d = Lowpassfilter.shape

        for x in range(0,d):
            for y in range(0,c):
                    distance = sqrt((x-d/2)**2 + (y-c/2)**2)
                    Lowpassfilter[y,x] = 1/(1+(distance/r)**(2*10))

        img = Fourier_transform(img)
        result = img * Inverse_filter * Lowpassfilter
        result1 = np.fft.ifft2(result)
        result2 = np.absolute(result1)

        max = np.amax(result2)
        result3 = (255 / max ) * result2
        result4 = np.array(result3, dtype=np.uint8)

        return result4
    ```
    * Wiener filtering
    ![Wiener_filter](https://user-images.githubusercontent.com/62092317/106533732-061aef80-6536-11eb-8ad8-fb6a4dee5434.PNG)
    ```python
    def Wiener_filter(img,k,K):
        Wiener_filter = np.zeros_like(img)
        Noise_filter = np.zeros_like(img)
        Noise_filter = np.array(Noise_filter, dtype=float)

        a, b = Noise_filter.shape

        for x in range(0, b):
            for y in range(0, a):
                Noise_filter[y, x] = (exp((-1)*k* ((y - b / 2) ** 2 + (x - a / 2) ** 2) ** (5 / 6)))
        Noise_abs = np.absolute(Noise_filter)
        Noise_abs_square  =np.square(Noise_abs)

        Wiener_filter = (1/Noise_filter)*(Noise_abs_square/(Noise_abs_square+K))
        img = Fourier_transform(img)
        result = img * Wiener_filter
        result1 = np.fft.ifft2(result)
        result2 = np.absolute(result1)

        max = np.amax(result2)
        result3 = (255 / max ) * result2
        result4 = np.array(result3, dtype=np.uint8)

        return result4
        
    def Wiener_with_Butterworth_filter(img,k,K,r):
        Wiener_filter = np.zeros_like(img)
        Noise_filter = np.zeros_like(img)
        Noise_filter = np.array(Noise_filter, dtype=float)

        a, b = Noise_filter.shape

        for x in range(0, b):
            for y in range(0, a):
                Noise_filter[y, x] = (exp((-1) * k * ((y - b / 2) ** 2 + (x - a / 2) ** 2) ** (5 / 6)))
        Noise_abs = np.absolute(Noise_filter)
        Noise_abs_square = np.square(Noise_abs)

        Wiener_filter = (1 / Noise_filter) * (Noise_abs_square / (Noise_abs_square + K))

        Lowpassfilter = np.zeros_like(img)
        Lowpassfilter = np.array(Lowpassfilter,dtype=float)
        c,d = Lowpassfilter.shape

        for x in range(0,d):
            for y in range(0,c):
                    distance = sqrt((x-d/2)**2 + (y-c/2)**2)
                    Lowpassfilter[y,x] = 1/(1+(distance/r)**(2*10))

        img = Fourier_transform(img)
        result = img * Wiener_filter * Lowpassfilter
        result1 = np.fft.ifft2(result)
        result2 = np.absolute(result1)

        max = np.amax(result2)
        result3 = (255 / max ) * result2
        result4 = np.array(result3, dtype=np.uint8)

        return result4
    ```
#See details in [HERE](https://github.com/SeongSuKim95/DIP_Various_Filter/blob/master/Spatial_and_Frequency_filter.pdf)

2020 MMI Lab. DIP 세미나 
Connected Component Labeling

1.	발표 내용 – https://scholar.google.com에서 논문을 검색
각 구현 내용을 함수나 class 별로 나눠서 함수의 재사용이 가능하도록 구현합니다. 이번 주차의 구현 내용은 다음과 같습니다.

1.Labeling
2가지 이상의 방법을 사용하여 connected component labeling을 구현합니다. 가능한 linked list를 이용하여 binary image에서 각각의 connected component에 대한 label을 부여합니다. 각 connected component를 다른 색으로 표시하고 각 label마다 포함된 pixel의 개수를 확인합니다. 

2. 공통사항
주어진 그림을 사용하여 올바른 결과가 나오는지 확인합니다. 또한 영상에 대해 연산량을 측정하고 최대한 효율적으로 구현합니다. 의미 있는 결과가 있다면 발표자료에 첨부합니다. 

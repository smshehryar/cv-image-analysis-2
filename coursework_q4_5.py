from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import os 


# Function to Convert Images to grayscale
def ICV_grayscale(img):

    height, width = img.shape[:2]     
    
    imgOutput= np.zeros(shape=(height,width))
    for i in range(height):
        for j in range(width):
           imgOutput[i, j] = (int(img[i, j, 0]*0.299)+int(img[i, j, 1]*0.587)+int(img[i, j, 2]*0.114))
           
    #plt.imshow(imgOutput)
    #plt.show()
    return imgOutput
#ICV_grayscale("Dataset/DatasetA/car-1.JPG")



############################################################
# START Question - 4 - Texture Descriptors and Classification
############################################################

#calculates the LBP values of a given image and its window size
#takes in the image name, start index x, start index y, window width and height, and if it is comparison
#OUTPUTS: the normalized window histogram and the LBP image
#RETURNS: the normalized window histogram to the method ICV_lbp_main()
def ICV_lbp_Image(fileName, indX, indY, window_width, window_height, compare):
    
    
    img = plt.imread(fileName)

    height, width = img.shape[:2] 

    img = ICV_grayscale(img) 
 
    imgOutput = np.zeros(shape=(width,height))
    imgOutputGray = np.zeros(shape=(width,height))
    

    window_hist = [0]*256
    image_bits = [0]*8
    
    #iterating over each pixel of the window
    for i in range(indX,window_height+indX):
        for j in range(indY,window_width+indY):
            

            #conditional statements below are set below to ignore the borders
            
            # for a 3x3 kernel the center pixel value is set as a threshold to compare the surrounding pixels
            # if the surrounding pixels are equal to or greater than the canter value they are given a bit value of
            # 1, otherwise 0. 
            center_val = img[i, j]
            
            #upper
            if(img[i-1,j] >= center_val):
                image_bits[0] = 1


            if(i+1<height):
                #lower
                if(img[i+1, j] >= center_val):
                    image_bits[4] = 1

                #lower left
                if(img[i+1, j-1] >= center_val):
                    image_bits[5] = 1

            
            #left
            if(img[i, j-1] >= center_val):
                image_bits[6] = 1

            
            if(j+1<width):
                #right
                if(img[i, j+1] >= center_val):
                    image_bits[2] = 1
                #upper right
                if(img[i-1, j+1]>= center_val):
                    image_bits[1] = 1
                
            
            if(j+1<width) and (i+1<height):
                #lower right
                if(img[i+1, j+1]>= center_val):
                    image_bits[3] = 1
                
            #upper left    
            if(img[i-1, j-1]>= center_val):
                    image_bits[7] = 1
            
            #each pixel of the output image is calculated by converting the binary number in the bit array into a decimal value.
            imgOutput[i,j] = int("".join(str(x) for x in image_bits),2)
            
            image_bits = [0,0,0,0,0,0,0,0]
            imgOutputGray[i,j] = img[i, j]
            
            #print(imgOutput[i, j])

            # color range values for the window histogram are updated here
            window_hist[int(imgOutput[i, j])] += 1
            
    

            
    sumWinHist = np.sum(window_hist)
    
    if(compare == 0):
        # loop to display the histogram
        for i in range(0, 256):
            plt.bar(i, (window_hist[i]/sumWinHist), color='black', alpha=0.8)

        plt.xlabel('Color Range (0-256)')
        plt.ylabel('Pixel Frequencies')
        plt.title('Plot for Color Range Frequencies of a Window') 
        plt.show()

        
        # show the LBP image window
        plt.imshow(imgOutput.astype('uint8'), cmap=plt.get_cmap('gray'))
        plt.show()
        # show the actual image window
        #plt.imshow(imgOutputGray.astype('uint8'), cmap=plt.get_cmap('gray'))
        #plt.show()

    for i in range(0, 256):
            window_hist[i] = window_hist[i]/sumWinHist

    return window_hist

#ICV_lbp_Image("Dataset/DatasetA/car-3.JPG")



#takes in the arguments of an image, 0 or 1 if it is comaparison, and the image division
#OUTPUTS: the global descriptor histogram of the whole image
#RETURNS: the global histogram to the method ICV_lbp_compare()
def ICV_lbp_main(fileName, compare, div):

    

    img = plt.imread(fileName)
    height, width = img.shape[:2]

    divisor = div

    window_width = int(width/divisor)
    window_height = int(height/divisor)

    indX = 0
    indY = 0
    total_hist = []
    
    # iterating for each window
    for i in range(divisor):
        for j in range(divisor):
          
          # each window histogram array is added to the array of the histogram of the complete image
          total_hist +=  ICV_lbp_Image(fileName, indX, indY, window_width, window_height, compare)
          indX += window_height
          
        indY += window_width
        indX = 0
    if(compare == 0):
        
        #loop to display the histogram of the total image
        for i in range(0, len(total_hist)):
            plt.bar(i, (total_hist[i]), color='black', alpha=0.8)
            plt.xlabel('Combined histogram ranges per 256 color range')
            plt.ylabel('Pixel Frequencies')
            plt.title('Plot for Concatenated Histograms of Complete Image')
        plt.show()    
    return total_hist,  np.sum(total_hist)


#ICV_lbp_main("Dataset/DatasetA/car-3.JPG",0,3)



# takes in the arguments of two images to be compared and the image divisions
#OUTPUTS: the label of the given image to be classified, against a second image which is kept as a comparison
def ICV_lbp_compare(fileName1, fileName2,div):

    #get the histograms of the two LBP images
    hist1, hist1Pixels = ICV_lbp_main(fileName1,1,div)
    hist2, hist2Pixels = ICV_lbp_main(fileName2,1,div)

    print(hist1Pixels)

    inter_hist = [0]*len(hist1)

    sum_hist = 0.0

    #intersection of normalized values
    for i in range(0,len(hist1)):
        inter_hist[i] = min(hist1[i]/hist1Pixels, hist2[i]/hist1Pixels)
        
        sum_hist += inter_hist[i]

    if(sum_hist >= 0.7267):
        print("Image classified as a Face")
        print("Percentage Match: ",(sum_hist*100))
    else:
        print("Image classified as a Car")
        print("Percentage Match: ",(sum_hist*100))
    

#ICV_lbp_compare("Dataset/DatasetA/face-2.JPG", "Dataset/DatasetA/car-2.JPG", 3)

############################################################
# END Question - 4 - Texture Descriptors and Classification
############################################################



############################################################
# START Question - 5 - Object Segmentation and Counting
############################################################

#takes the absolute difference between two frames
#OUTPUTS: the threshold image
def ICV_frame_differencing(frame0, frame1):

    img0 = plt.imread(frame0) 
    img1 = plt.imread(frame1)

    height, width = img0.shape[:2]

    #img0 = grayscale(frame0)
    #img1 = grayscale(frame1)

    width = width
    height = height

    #imgOutput = np.zeros(shape=(height,width))
    imgOutput = np.ndarray((height,width,3))

    for i in range(height):
        for j in range(width):

            r0 = int(img0[i,j,0])
            g0 =  int(img0[i,j,1])
            b0 =  int(img0[i,j,2])
            r1 =  int(img1[i,j,0])
            g1 =  int(img1[i,j,1])
            b1 =  int(img1[i,j,2])

            #taking the difference of the RGB values
            diff = (abs(r0-r1)+abs(g0-g1)+abs(b0-b1))/3

            #if the difference is below a certain threshold the pixel values are assigned a color of black, otherwise white
            if(diff < 20):
                imgOutput[i,j,0] = 0
                imgOutput[i,j,1] = 0
                imgOutput[i,j,2] = 0
            else:
                imgOutput[i,j,0] = 255
                imgOutput[i,j,1] = 255
                imgOutput[i,j,2] = 255
                
            
        
    #plt.imshow(imgOutput.astype('uint8'), cmap=plt.get_cmap('gray'))
    plt.imshow(imgOutput.astype('uint8'))
    plt.show()

#ICV_frame_differencing("frame0.JPG", "frame5.JPG")




# takes the average over a given no of frames
#OUTPUTS: the generated background 
def ICV_background_estimate(noOfFrames):

    no_of_frames = noOfFrames

    imgTemp = plt.imread('frame'+str(no_of_frames)+'.jpg')
    height, width = imgTemp.shape[:2]
    
    
    imgOutput = np.ndarray((height,width,3))
    imgOutputTemp = np.ndarray((height,width,3))

    #iterating over the number of frames to be averaged
    for index in range(no_of_frames):
        
        imgTemp = plt.imread('frame'+str(index)+'.jpg')

        height, width = imgTemp.shape[:2]
        
        # iterating over each pixel for every frame
        for i in range(height):
            for j in range(width):
                
                #Each image channel is summed up in a temporary array
                imgOutputTemp[i,j,0] += imgTemp[i,j,0]
                imgOutputTemp[i,j,1] += imgTemp[i,j,1]
                imgOutputTemp[i,j,2] += imgTemp[i,j,2]
    
    # To assign the averaged result to the output image, each summed up channel is divided by the number of frames
    for i in range(height):
        for j in range(width):
            imgOutput[i,j,0] = imgOutputTemp[i,j,0]/(no_of_frames)
            imgOutput[i,j,1] = imgOutputTemp[i,j,1]/(no_of_frames)
            imgOutput[i,j,2] = imgOutputTemp[i,j,2]/(no_of_frames)

    
    #cv2.imshow("show",imgOutput)
    #cv2.imwrite('background.jpg',imgOutput)

    #fig = plt.figure(frameon=False)
    #fig = plt.figure(figsize=(width/120, height/120), dpi=120)
    #ax = plt.Axes(fig, [0., 0., 1., 1.])
    #ax.set_axis_off()
    #fig.add_axes(ax)
    #ax.imshow(imgOutput.astype('uint8'), aspect='auto')
    
    #fig.savefig('background_50.jpg',bbox_inches='tight',transparent=True, pad_inches=0)

    
    plt.imshow(imgOutput.astype('uint8'))
    plt.show()

#ICV_background_estimate(50)



############### References
#https://www.youtube.com/watch?v=ce-2l2wRqO8 - Blob detection

# class defined to create blob objects
class Blob:
    minx = 0
    miny = 0
    maxx = 0
    maxy = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 1
        self.height = 1

        self.minx = x
        self.miny = y
        self.maxx = x
        self.maxy = y

    #calculates the distance between two points
    #returns a boolean to indicate if distance is above or below a certain threshold
    def ICV_proximity(self,x,y):
        centerX = (self.minx + self.maxx)/2
        centerY = (self.miny + self.maxy)/2

        diff = (abs(centerX-x) + abs(centerY-y))/2
        #print(diff)

        if( diff < 40):
            
            return True
        else:
            return False

    # adds a coordinate point to the number of points a blob contains
    def ICV_addPoint(self,x,y):

        self.minx = min(self.minx,x)
        self.miny = min(self.miny,y)
        self.maxx = max(self.maxx,x)
        self.maxy = max(self.maxy,y)

    # draws a rectangular area on the image where blobs are cerated
    def ICV_drawArea(self, imgTemp, colorDiff):

        height, width = imgTemp.shape[:2]
        
        
        # iterating over each pixel for every frame
        for i in range(height):
            for j in range(width):

                if( (self.minx < i < self.maxx) and (self.miny < j < self.maxy)):

                    imgTemp[i,j] = colorDiff

        plt.imshow(imgTemp.astype('uint8'), cmap=plt.get_cmap('gray'))
        plt.show()




def ICV_filter_Image(fileName, gray, w, h):
    
   
    filterDilate = np.array([
        [0,1,0],
        [1,1,1],
        [0,1,0]])
    
    Tfilter = filterDilate
    
    img = fileName
    #if(gray == 1):
    #    img = ICV_grayscale(img)
    width = w
    height = h

    imgOutput = np.zeros(shape=(height,width))
    
    upper = 0
    lower = 0
    right = 0
    left = 0

    if(gray == 1):
        for i in range(height):
            for j in range(width):
                
                
                if(i-1>height):
                    upper = img[i-1, j] or (Tfilter[0,1])

                if(i+1<height):
                    lower = img[i+1, j] or (Tfilter[2,1])
                    lower_left = img[i+1, j-1] or (Tfilter[2,0])

                if(j+1<width):
                    right = img[i, j+1] or (Tfilter[1,2])
                    upper_right= img[i-1, j+1] or (Tfilter[0,2])
                
                if(j+1<width) and (i+1<height):
                    lower_right = img[i+1, j+1] or (Tfilter[2,2])

                left = img[i, j-1]*(Tfilter[1,0])
                upper_left = img[i-1, j-1] or (Tfilter[0,0])
                

                imgOutput[i, j] = ( upper+lower+left+right+lower_left+lower_right+upper_left+upper_right)
                #imgOutput[i, j] = (img[i, j] or (Tfilter[1,1]))
                
     
    #plt.axis('off')
            
    #plt.imshow(imgOutput.astype('uint8'), cmap=plt.get_cmap('gray'))
    #plt.savefig("test.jpg", bbox_inches='tight') 
    #plt.show()

    return imgOutput

        
  
# segments moving obejects from a sequence of frames given a background reference frame
#OUTPUTS: the number of objects detected
def ICV_segment(background):

    

    no_of_frames = 1

    imgTemp = plt.imread('frame'+str(no_of_frames)+'.jpg')
    height, width = imgTemp.shape[:2]

    imgBackground = plt.imread(background)
    
    height, width = imgBackground.shape[:2]
    imgBackground = ICV_grayscale(imgBackground)
    
    imgOutput = np.ndarray((height,width,3))
    #imgOutputTemp = np.ndarray((height,width,3))
    imgOutputTemp = np.zeros(shape=(height,width))

    blobArray = []
    blobCountArray = [0]*no_of_frames

    #iterating over the number of frames to be averaged
    for index in range(no_of_frames):
        
        imgTemp = plt.imread('frame'+str(index)+'.jpg')
        
        height, width = imgTemp.shape[:2]
        imgTemp = ICV_grayscale(imgTemp)
        
        # iterating over each pixel for every frame
        for i in range(height):
            for j in range(width):
                
                r0 = int(imgBackground[i,j])
                r1 =  int(imgTemp[i,j])

            
                diff = (abs(r0-r1))/1
                #print(diff)
                if(diff < 20):
                #Each image channel is summed up in a temporary array
                    imgOutputTemp[i,j] = 0

                else:
                #Each image channel is summed up in a temporary array
                    imgOutputTemp[i,j] = 255


        # getting the dilated image
        imgOutputTemp = ICV_filter_Image(imgOutputTemp,1,width,height)
        
        countBlob = 0
        blobArray = []
        # iterate over the dilated image to detect blobs
        for i in range(height):
            for j in range(width):

                if(imgOutputTemp[i,j] >= 200):
                    
                    found = False
                    for blob in blobArray:
                        # checking the distance from each blob
                        if(blob.ICV_proximity(i,j)):
                            # if the distance is below a threshold add the point to the blobs dimensions
                            blob.ICV_addPoint(i,j)
                            found = True
                            break
                    # if blob not found create a new blob     
                    if(found == False):

                        bTemp = Blob(i,j)
                        blobArray.append(bTemp)
                        #print(i,j)
                        

         
        colorDiff = 20
        
        for blob in blobArray:
            #calculate the area of each blob
            area = (blob.maxx- blob.minx)*(blob.maxy-blob.miny)
            # apply area threshold
            if(area > 5000):
                countBlob +=1
                colorDiff += 40
                #blob.ICV_drawArea(imgOutputTemp, colorDiff)

        # add the blob count per frame into respective frame index
        blobCountArray[index] = countBlob

        #plt.imshow(imgOutputTemp.astype('uint8'), cmap=plt.get_cmap('gray'))
        #plt.show()

    for i in range(0, len(blobCountArray)):
        plt.bar(i, (blobCountArray[i]), color='black', alpha=0.8)
        plt.xlabel('Frame count')
        plt.ylabel('Number of Objects')
        plt.title('Plot for Number of Objects per frame')
    plt.show()    

ICV_segment("background.jpg")


############################################################
# END Question - 5 - Object Segmentation and Counting
############################################################



############### References
#https://www.youtube.com/watch?v=ce-2l2wRqO8 - Blob detection

############### README

# To run a function uncomment the function call

############### Texture Descriptors and Classification
############################################################

#### FUNCTION 1 - calculate LBP values
#calculates the LBP values of a given image and its window size
#takes in the image name, start index x, start index y, window width and height, and if it is comparison
#OUTPUTS: the normalized window histogram and the LBP image
#RETURNS: the normalized window histogram to the method ICV_lbp_main()

#ICV_lbp_Image(fileName, window start X, window start Y, window_width, window_height, to compare?)
#to compare?: 0=no, 1=yes

#ICV_lbp_Image("Dataset/DatasetA/car-3.JPG", 0, 0, 71, 71, 0)

#### FUNCTION 2 - divide imge into windows
#takes in the arguments of an image, 0 or 1 if it is comaparison, and the image division
#OUTPUTS: the global descriptor histogram of the whole image
#RETURNS: the global histogram to the method ICV_lbp_compare()

#ICV_lbp_main(fileName, to compare?, window division in one dimension)
#to compare?: 0=no, 1=yes
#window division in one dimension: 3, if image needs to be divided into 9 windows

#ICV_lbp_main("Dataset/DatasetA/car-3.JPG", 0, 3)

#### FUNCTION 3 - compare images
# takes in the arguments of two images to be compared and the image divisions
#OUTPUTS: the label of the given image to be classified, against a second image which is kept as a comparison

#ICV_lbp_compare(image to classified, reference image, window division in one dimension)
#window division in one dimension: 3, if image needs to be divided into 9 windows

#ICV_lbp_compare("Dataset/DatasetA/face-2.JPG", "Dataset/DatasetA/car-3.JPG",3)


############### Object Segmentation and Counting
############################################################

#### FUNCTION 1 - calculates the difference between frames
#takes the absolute difference between two frames
#OUTPUTS: the threshold image

#ICV_frame_differencing(frame0, frame1)


#### FUNCTION 2 - estimates the background

# takes the average over a given no of frames
#OUTPUTS: the generated background

#ICV_background_estimate(noOfFrames)

#### FUNCTION 3 - segments the moving objects from the background

# segments moving obejects from a sequence of frames given a background reference frame
#OUTPUTS: the number of objects detected

#ICV_segment(background)
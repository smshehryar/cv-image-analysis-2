
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
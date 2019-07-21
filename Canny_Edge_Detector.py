import math
from PIL import Image
import numpy as np

#PIL library used for read and write image functions
#numpy library used for array objects and array copy function numpy.zeros_like(array)

#All functions work for 8 bit grayscale image bmps (one color value at each pixel)

def convertBMPToIntensity(imgArray):
	#Converts From RGB to Intensity Using unweighted average
	convertedImg = np.zeros((imgArray.shape[0],imgArray.shape[1]),dtype = int)

	for i in range(len(convertedImg)):
		for j in range(len(convertedImg[i])):
			convertedImg[i][j] = (imgArray[i][j][0] + imgArray[i][j][1] + imgArray[i][j][2])/3

	return convertedImg

def openBMPAsArray(fileName):

	#Function that opens a BMP file  and returns it as a numpy array

	im = Image.open(fileName)
	a = np.array(im)
	a = a.astype(int)

	return a

def saveArrayAsBMP(imgArray,filename):

	# Converts threshold array in numpy array form to 8 bit grayscale bmp and saves it with input file name
	im = Image.fromarray(np.uint8(imgArray))
	im.save(filename)


def sumOfArrayElems(a):

	#Function used to sum elements in 2d array

	sumOfElem = 0

	for i in range(len(a)):
		for j in range(len(a[i])):
			sumOfElem += a[i][j]

	return sumOfElem


def gaussSmoothingConvolution(imgArray, mask, i, j):

	#This function returns the weighted sum of the integers in the neighborhood of the pixel at row index i and column index j
	#The function calculates the center of a symmetric mask that way a mask of variable size can be used

	maskCenterInd = math.ceil(len(mask)/2)-1
	weightedSum = 0
	for k in range(-maskCenterInd,maskCenterInd+1):
		for l in range(-maskCenterInd,maskCenterInd+1):
			weightedSum += (imgArray[i+k][j+l])*(mask[k+maskCenterInd][l+maskCenterInd])

	return weightedSum

def gaussianSmoothing(imgArray):

	#Create array with zero values with shape of image array
	smoothedArray = np.zeros_like(imgArray)

	#Mask for 7xy Gaussian array
	mask = np.array(
		[[1,1,2,2,2,1,1],
		[1,2,2,4,2,2,1],
		[2,2,4,8,4,2,2],
		[2,4,8,16,8,4,2],
		[2,2,4,8,4,2,2],
		[1,2,2,4,2,2,1],
		[1,1,2,2,2,1,1]]
		)

	#Calculate Center of Mask
	maskCenterInd = math.ceil(len(mask)/2)-1

	
	#Value of -9999 is used to represent undefined values in the smoothed array
	#If part of the mask goes outside the boundaries of the image the value of the smoothed array
	#at coordinate i,j will be undefined (-9999)
	for i in range(len(imgArray)):
		for j in range(len(imgArray[i])):
			if ( i-maskCenterInd < 0 or i+maskCenterInd >= len(imgArray)):
				smoothedArray[i][j] = -9999
			elif ( j-maskCenterInd < 0 or j+maskCenterInd >=len(imgArray[i])):
				smoothedArray[i][j] = -9999
			else:
				smoothedArray[i][j] = gaussSmoothingConvolution(imgArray,mask, i,j)
				smoothedArray[i][j] = smoothedArray[i][j]/sumOfArrayElems(mask)
			
	return smoothedArray


def prewittGradientConvolution(imgArray, mask, i, j):

	#This function returns the weighted sum of the integers in the neighborhood of the pixel at row index i and column index j
	#The function calculates the center of a symmetric mask that way a mask of variable size can be used
	maskCenterInd = math.ceil(len(mask)/2)-1
	weightedSum = 0
	for k in range(-maskCenterInd,maskCenterInd+1):
		for l in range(-maskCenterInd,maskCenterInd+1):
			#-9999 represents an undefined value.
			#If function encounters this value, it will return 0.
			if imgArray[i+k][j+l] == -9999:
				return 0
			else:
				weightedSum += (imgArray[i+k][j+l])*(mask[k+maskCenterInd][l+maskCenterInd])

	return weightedSum


def prewittGradientXDir(imgArray):

	xMask = np.array(
		[[-1,0,1],
		[-1,0,1],
		[-1,0,1]]
		)

	maskCenterInd = math.ceil(len(xMask)/2)-1

	gradientX = np.zeros_like(imgArray)

	#If part of the mask goes into the undefined region an edge value of zero will be returned
	#at that i,j coordinate
	for i in range(len(imgArray)):
		for j in range(len(imgArray[i])):
			if (i-maskCenterInd < 0 or i+maskCenterInd >= len(imgArray)):
				gradientX[i][j] = 0
			elif (j-maskCenterInd < 0 or j+maskCenterInd >= len(imgArray[i])):
				gradientX[i][j] = 0
			else:
				#Maximum absolute value of unnormalized x gradient is 3*255
				#Divide calculated x gradient by 3 to normalize
				#I did not take absolute value at this point to preserve direction for gradient angle calculation
				gradientX[i][j] = prewittGradientConvolution(imgArray, xMask, i, j)/3
				#gradientX[i][j] = abs(prewittGradientConvolution(imgArray, xMask, i, j))/3

	return gradientX


def prewittGradientYDir(imgArray):

	yMask = np.array(
		[[1,1,1],
		[0,0,0],
		[-1,-1,-1]]
		)

	maskCenterInd = math.ceil(len(yMask)/2)-1

	gradientY = np.zeros_like(imgArray)
	#If part of the mask goes into the undefined region an edge value of zero will be returned
	#at that i,j coordinate
	for i in range(len(imgArray)):
		for j in range(len(imgArray[i])):
			if (i-maskCenterInd < 0 or i+maskCenterInd >= len(imgArray)):
				gradientY[i][j] = 0
			elif (j-maskCenterInd < 0 or j+maskCenterInd >= len(imgArray[i])):
				gradientY[i][j] = 0
			else:
				#Maximum absolute value of unnormalized y gradient is 3*255
				#Divide calculated y gradient by 3 to normalize
				#I did not take absolute value at this point to preserve direction for gradient angle calculation
				gradientY[i][j] = prewittGradientConvolution(imgArray, yMask, i, j)/3
				

	return gradientY

def gradientMagnitude(gradientX,gradientY):

	magnitude = np.zeros_like(gradientX)

	for i in range(len(magnitude)):
		for j in range(len(magnitude[i])):
			magnitude[i][j] = math.sqrt(gradientX[i][j]**2 + gradientY[i][j]**2)/math.sqrt(2)

	return magnitude

def gradientDirection(gradientX,gradientY):

	direction = np.zeros_like(gradientX)

	for i in range(len(direction)):
		for j in range(len(direction[i])):
			if (gradientX[i][j] == 0 and gradientY[i][j] == 0):
				direction[i][j] = 0
			elif (gradientX[i][j] == 0):
				direction[i][j] = 90
			else:
				direction[i][j] = math.degrees(math.atan(gradientY[i][j]/gradientX[i][j]))
				if(direction[i][j] < 0):
					direction[i][j] += 360

	return direction

def calculateQuantizedAngle(gradientAngle):
	#Takes in gradient angle values and maps to quantized angle sector value for non-maxima suppression
	quantizedAngle = np.zeros_like(gradientAngle)
	for i in range(len(quantizedAngle)):
		for j in range(len(quantizedAngle[i])):
			if (gradientAngle[i][j] >=0 and gradientAngle[i][j] < 22.5):
				quantizedAngle[i][j] = 0
			elif (gradientAngle[i][j] >= 22.5 and gradientAngle[i][j] < 67.5):
				quantizedAngle[i][j] = 1
			elif (gradientAngle[i][j] >= 67.5 and gradientAngle[i][j] < 112.5):
				quantizedAngle[i][j] = 2
			elif (gradientAngle[i][j] >= 112.5 and gradientAngle[i][j] < 157.5):
				quantizedAngle[i][j] = 3
			elif (gradientAngle[i][j] >= 157.5 and gradientAngle[i][j] < 202.5):
				quantizedAngle[i][j] = 0
			elif (gradientAngle[i][j] >= 202.5 and gradientAngle[i][j] < 247.5):
				quantizedAngle[i][j] = 1
			elif (gradientAngle[i][j] >= 247.5 and gradientAngle[i][j] < 292.5):
				quantizedAngle[i][j] = 2
			elif (gradientAngle[i][j] >= 292.5 and gradientAngle[i][j] < 337.5):
				quantizedAngle[i][j] = 3
			elif (gradientAngle[i][j] >= 337.5 and gradientAngle[i][j] < 360):
				quantizedAngle[i][j] = 0

	return quantizedAngle

def nmsCompareToNeighbors(gradientMagnitude,quantizedAngle,i,j):

	#Helper function to determine which neighbors to compare to in non-maxima suppression based on sector value

	if quantizedAngle[i][j] == 0:
		if((gradientMagnitude[i][j] <= gradientMagnitude[i][j+1]) or (gradientMagnitude[i][j] <= gradientMagnitude[i][j-1])):
			return 0
	elif quantizedAngle[i][j] == 1:
		if((gradientMagnitude[i][j] <= gradientMagnitude[i-1][j+1]) or (gradientMagnitude[i][j] <= gradientMagnitude[i+1][j-1])):
			return 0
	elif quantizedAngle[i][j] == 2:
		if((gradientMagnitude[i][j] <= gradientMagnitude[i-1][j]) or (gradientMagnitude[i][j] <= gradientMagnitude[i+1][j])):
			return 0
	elif quantizedAngle[i][j] == 3:
		if((gradientMagnitude[i][j] <= gradientMagnitude[i-1][j-1]) or (gradientMagnitude[i][j] <= gradientMagnitude[i+1][j+1])):
			return 0


	return gradientMagnitude[i][j]


def nonMaximaSuppression(gradientMagnitude, quantizedAngle):

	
	nmsGradMagnitude = np.zeros_like(gradientMagnitude)
	for i in range(len(gradientMagnitude)):
		for j in range(len(gradientMagnitude[i])):
			if (gradientMagnitude[i][j] != 0):
				nmsGradMagnitude[i][j] = nmsCompareToNeighbors(gradientMagnitude,quantizedAngle,i,j)

	return nmsGradMagnitude



def pTileMethod(imgArray, percentage):

	#Input is image array and percentage of pixels that should have normalized value greater than or equal to threshold
	#Percentage input value is in % (Ex: Input 10 for 10%)
	grayScaleCount = np.zeros(shape = (256),dtype = int)
	
	for i in range(len(imgArray)):
		for j in range(len(imgArray[i])):
			if (imgArray[i][j] !=0):
				grayScaleCount[(imgArray[i][j])] += 1

	
	totalEdges = 0
	for i in range(len(grayScaleCount)):
		totalEdges+=grayScaleCount[i]



	foregroundCount = int(round((totalEdges*percentage)/100))
	backgroundCount = int(round((totalEdges*(100-percentage))/100))
	
	
	#For the P Tile Method, we must iterate from both sides of the histogram and compare the two results
	#to compute the most accurate value of T

	index = 255
	
	tempCount1 = foregroundCount
	actualForegroundCount = 0
	temp1 = index
	while(tempCount1 > 0):
		actualForegroundCount += grayScaleCount[index]
		temp1 = index
		tempCount1 -= grayScaleCount[index]
		index -= 1
	

	actualBackgroundCount = 0

	index = 0
	tempCount2 = backgroundCount
	temp2 = 1
	while(tempCount2 > 0):
		actualBackgroundCount += grayScaleCount[index]
		temp2 = index+1
		tempCount2 -= grayScaleCount[index]
		index += 1


	if abs((actualForegroundCount/totalEdges) - percentage/100) < abs((totalEdges-actualBackgroundCount)/totalEdges - percentage/100):
		threshold = temp1
	else:
		threshold = temp2


	thresholdImage = np.zeros_like(imgArray)

	for i in range(len(thresholdImage)):
		for j in range(len(thresholdImage[i])):
			if(imgArray[i][j]<threshold):
				thresholdImage[i][j] = 0
			else:
				thresholdImage[i][j] = 255

	print("Threshold is ", threshold)
	return thresholdImage



def countEdges(thresholdImage):

	edgeCount = 0
	for i in range(len(thresholdImage)):
		for j in range(len(thresholdImage[i])):
			if thresholdImage[i][j] != 0:
				edgeCount+=1

	return edgeCount


if __name__ == '__main__':

	#Prompt user for file name
	print("Please input name of bmp file with extension:")

	filename = input()

	#Open image file and save as a numpy array for processing
	#Example input if files are in same directory : lena256.bmp

	imgArray = openBMPAsArray(filename)

	if len(imgArray.shape) > 2:
		imgArray = convertBMPToIntensity(imgArray)


	#Smooth Array using gaussian smoothing. Normalize values by dividing each pixel value by the sum of 
	#the integers in the mask
	smoothedArray = gaussianSmoothing(imgArray)


	#Use the Prewitt Gradient Operator to calculate the gradient
	#Calculate the x and y gradient responses
	gradientX = prewittGradientXDir(smoothedArray)
	gradientY = prewittGradientYDir(smoothedArray)

	#Using the x and y gradient calculate gradient magnitude and direction
	gradMagnitude = gradientMagnitude(gradientX,gradientY)
	gradAngle = gradientDirection(gradientX,gradientY)

	#Calculate quantized angle for non-maxima suppression
	quantizedAngle = calculateQuantizedAngle(gradAngle)
	#Use non-maxima suppression to suppress all edge values that are not local maxima
	nmsGradMagnitude = nonMaximaSuppression(gradMagnitude,quantizedAngle)

	#Use P-Tile Thresholding to produce thresholded output for P= 10%, 30%, and 50%
	#Edge values greater than or equal to threshold will be set to 255 and all values less than threshold
	#will be set to 0
	print("For P = 10")
	thresholdImage10Percent = pTileMethod(nmsGradMagnitude,10)
	print("Number of Edges: ",countEdges(thresholdImage10Percent))
	print("")
	print("For P = 30")
	thresholdImage30Percent = pTileMethod(nmsGradMagnitude,30)
	print("Number of Edges: ",countEdges(thresholdImage30Percent))
	print("")
	print("For P = 50")
	thresholdImage50Percent = pTileMethod(nmsGradMagnitude,50)
	print("Number of Edges: ",countEdges(thresholdImage50Percent))
	

	#Save image outputs as 8 bit bmp files
	saveArrayAsBMP(smoothedArray, "SmoothedImage.bmp")


	#Before saving x and y gradient responses take absolute value at each point
	#Absolute value was not taken earlier to preserve direction for gradient angle calculation
	for i in range(len(gradientX)):
		for j in range(len(gradientX[i])):
			gradientX[i][j] = abs(gradientX[i][j])
	saveArrayAsBMP(gradientX, "XPrewittGradient.bmp")

	for i in range(len(gradientY)):
		for j in range(len(gradientY[i])):
			gradientY[i][j] = abs(gradientY[i][j])
	saveArrayAsBMP(gradientY,"YPrewittGradient.bmp")

	saveArrayAsBMP(gradMagnitude, "GradientMagnitude.bmp")
	saveArrayAsBMP(nmsGradMagnitude, "NMSGradMagnitude.bmp")
	saveArrayAsBMP(thresholdImage10Percent,"OutputImage10percent.bmp")
	saveArrayAsBMP(thresholdImage30Percent,"OutputImage30percent.bmp")
	saveArrayAsBMP(thresholdImage50Percent,"OutputImage50percent.bmp")
	

	
	




	



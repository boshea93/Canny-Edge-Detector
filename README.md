# Canny-Edge-Detector
<h1>Project for Creating a Canny Edge Detector in Python (Input File Format bmp)</h1>

<h2>About</h2>
The purpose of this assignment was to create a canny edge detector in Python. There are 4 main steps in producing our final results. First,
Gaussian smoothing is applied to the input image to reduce noise. Next the gradient of the image is computed using the Prewitt operator.
After the gradient has been computed, non-maxima suppression is used to create thinner edges in our result images. The last step of the 
process is to use P-Tile Thresholding to filter a given percentage of the remaining edges, based on gradient magnitude. Input images should
be in bmp format. The script outputs 8 images from each step of the script: gaussian blurred image, x-gradient, y-gradient, gradient 
magnitude, gradient magnitude after non-maxima suppression, and three output images with different percentages of edges remaining after
p-tile thresholding. 

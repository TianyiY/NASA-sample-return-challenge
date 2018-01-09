# Import some packages from matplotlib
# Import the "numpy" package for working with arrays
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Define the filename, read and plot the image
filename = 'sample.jpg'
image = mpimg.imread(filename)

print(image.dtype, image.shape, np.min(image), np.max(image))
# uint8 (160, 320, 3) 0 255
# 8-bit unsigned integer array (uint8), the size of the array is (160, 320, 3)
# the minimum and maximum values are 0 and 255, respectively.
# This comes from the fact that with 8 bits of information for each pixel in each color channel,
# you have 2^​8​​ or 256 possible values, with the minimum possible value being 0 and the maximum being 255.
# Not all images are scaled this way so it's always a good idea to check the range and data type of the array after reading in an image if you're not sure.

plt.imshow(image)
plt.show()

# Note: we use the np.copy() function rather than just saying red_channel = image
# because in Python, such a statement would set those two arrays equal to each other
# forever, meaning any changes made to one would also be made to the other!
red_channel = np.copy(image)
# Note: here instead of extracting individual channels from the image
# I'll keep all 3 color channels in each case but set the ones I'm not interested
# in to zero.
red_channel[:,:,[1, 2]] = 0 # Zero out the green and blue channels
green_channel = np.copy(image)
green_channel[:,:,[0, 2]] = 0 # Zero out the red and blue channels
blue_channel = np.copy(image)
blue_channel[:,:,[0, 1]] = 0 # Zero out the red and green channels
fig = plt.figure(figsize=(12,3)) # Create a figure for plotting
plt.subplot(131) # Initialize subplot number 1 in a figure that is 3 columns 1 row
plt.imshow(red_channel) # Plot the red channel
plt.subplot(132) # Initialize subplot number 2 in a figure that is 3 columns 1 row
plt.imshow(green_channel)  # Plot the green channel
plt.subplot(133) # Initialize subplot number 3 in a figure that is 3 columns 1 row
plt.imshow(blue_channel)  # Plot the blue channel
plt.show()

'''
the mountains are relatively dark (low intensity values) in all three color channels, 
both the ground and the sky are brighter (higher intensity) in the red, green and blue channels. 
However, in all cases it looks like the ground is a bit brighter than the sky, 
such that it should be possible to identify pixels associated with the ground using a simple color threshold. 

to write a function that takes as its input a color image and a 3-tuple of color threshold values (integer values from 0 to 255), 
and outputs a single-channel binary image, which is to say an image with a single color channel where every pixel is set to one or zero. 
In this case, all pixels that were above the threshold should be assigned a value of 1, and those below a value of 0
'''


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np

image = mpimg.imread('grid.jpg')
plt.imshow(image)
plt.show()

def perspect_transform(image, source, destination):
    # use cv2.getPerspectiveTransform() to get Map, the transform matrix.
    Map = cv2.getPerspectiveTransform(source, destination)
    # use cv2.warpPerspective() to apply Map and warp your image to a top-down view.
    # keep same size as input image
    warped = cv2.warpPerspective(image, Map, (image.shape[1], image.shape[0]))
    # return the transformed image
    return warped

# define calibration box in source (actual) and destination (desired) coordinates
# these source and destination points are defined to warp the image to a grid where each 10x10 pixel square represents 1 square meter
destination_size = 5
# set a bottom offset to account for the fact that the bottom of the image is not the position of the rover but a bit in front of it
bottom_offset = 6
# define 4 source points, in this case, the 4 corners of a grid cell in the original image.
source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
# define 4 destination points (must be listed in the same order as source points).
destination = np.float32([[image.shape[1]/2 - destination_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + destination_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + destination_size, image.shape[0] - 2*destination_size - bottom_offset],
                  [image.shape[1]/2 - destination_size, image.shape[0] - 2*destination_size - bottom_offset]])

warped = perspect_transform(image, source, destination)
# draw Source and destination points on images (in blue) before plotting
cv2.polylines(image, np.int32([source]), True, (0, 0, 255), 3)
cv2.polylines(warped, np.int32([destination]), True, (0, 0, 255), 3)
# display the original image and binary image
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6), sharey=True)
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=40)

ax2.imshow(warped, cmap='gray')
ax2.set_title('Transformed Image', fontsize=40)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
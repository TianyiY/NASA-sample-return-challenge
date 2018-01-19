import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc      # for saving images
import cv2      # open cv
import glob     # for reading images
import imageio
imageio.plugins.ffmpeg.download()


images_path='IMG/*'
images=glob.glob(images_path)
# randomly display one image in IMG folder
index=np.random.randint(0, len(images)-1)
image_show=mpimg.imread(images[index])
plt.imshow(image_show)
plt.show()

# calibration
grid_image_path='CALIBRATION/grid1.jpg'
rock_image_path='CALIBRATION/rock1.jpg'
grid_image=mpimg.imread(grid_image_path)
rock_image=mpimg.imread(rock_image_path)
plt.subplot(211)
plt.imshow(grid_image)
plt.subplot(212)
plt.imshow(rock_image)
plt.show()

# Perspective Transform
# Define a function to perform a perspective transform
def perspect_transform(image, source, destination):
    Map = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(image, Map, (image.shape[1], image.shape[0]))  # keep same size as input image
    return warped

# Define calibration box in source (actual) and destination (desired) coordinates
destination_size = 5
# Set a bottom offset to account for the fact that the bottom of the image is not the position of the rover but a bit in front of it
bottom_offset = 6
source = np.float32([[14, 140], [301, 140], [200, 96], [118, 96]])
destination = np.float32([[image_show.shape[1] / 2 - destination_size, image_show.shape[0] - bottom_offset],
                          [image_show.shape[1] / 2 + destination_size, image_show.shape[0] - bottom_offset],
                          [image_show.shape[1] / 2 + destination_size, image_show.shape[0] - 2 * destination_size - bottom_offset],
                          [image_show.shape[1] / 2 - destination_size, image_show.shape[0] - 2 * destination_size - bottom_offset],
                          ])
warped_grid = perspect_transform(grid_image, source, destination)
warped_rock = perspect_transform(rock_image, source, destination)

plt.imshow(warped_grid)
scipy.misc.imsave('OUTPUT_IMAGES/warped_grid.jpg', warped_grid)
plt.show()

# Color Thresholding to find path and obstacle
# pick pixels above the threshold and set to 1, threshold of RGB set to 160
def color_threshold(image, RGB_threshold=(160, 160, 160)):
    # Create an array of zeros with the same size as image, but single channel
    Path = np.zeros_like(image[:, :, 0])
    index_above_threshold = (image[:, :, 0] > RGB_threshold[0]) & (image[:, :, 1] > RGB_threshold[1]) & (image[:, :, 2] > RGB_threshold[2])
    # set the pixels above RGB_threshold to 1
    Path[index_above_threshold] = 1
    Obs = np.ones_like(image[:, :, 0])
    Obs[index_above_threshold] = 0

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    obstacle = np.bitwise_and(Obs, image_gray)
    # Return the binary image
    return Path, obstacle

# finding rocks by color filtering
def rock_positioning(image, rock_RGB_threshold=(100, 100, 80)):
    rocks=np.zeros_like(image[:, :, 0])
    index_above_threshold = (image[:, :, 0] > rock_RGB_threshold[0]) & (image[:, :, 1] > rock_RGB_threshold[1]) & (image[:, :, 2] < rock_RGB_threshold[2])
    rocks[index_above_threshold]=1
    return rocks

warped = perspect_transform(rock_image, source, destination)
Path, _ = color_threshold(warped)
rocks = rock_positioning(warped)
plt.imshow(Path, cmap='gray')
plt.title('Path')
scipy.misc.imsave('OUTPUT_IMAGES/Path.jpg', Path)
plt.show()
plt.imshow(rocks, cmap='gray')
plt.title('Rocks')
scipy.misc.imsave('OUTPUT_IMAGES/Rocks.jpg', rocks)
plt.show()


# Coordinate Transformations
# Define a function to convert from image coords to rover coords
def to_rover_coords(binary_image):
    # get path positions (nonzero pixels)
    y_pos, x_pos = binary_image.nonzero()
    # Calculate pixel positions with reference to the rover position being at the center bottom of the image.
    x = -(y_pos - binary_image.shape[0]).astype(np.float)
    y = -(x_pos - binary_image.shape[1] / 2).astype(np.float)
    return x, y


# Define a function to convert to radial coords in rover space
def to_polar_coords(x, y):
    # Convert (x, y) to (radius, angle) in polar coordinates in rover space
    radius = np.sqrt(x ** 2 + y ** 2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y, x)
    return radius, angles

# rotation and translation
def rotate_coords(x, y, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180.
    x_rotated = (x * np.cos(yaw_rad)) - (y * np.sin(yaw_rad))
    y_rotated = (x * np.sin(yaw_rad)) + (y * np.cos(yaw_rad))
    return x_rotated, y_rotated

def translate_coords(x_rotated, y_rotated, x_pos, y_pos, scale):
    # Apply a scaling and a translation
    x_translated = (x_rotated / scale) + x_pos
    y_translated = (y_rotated / scale) + y_pos
    # Return the result
    return x_translated, y_translated


# Define a function to map rover space pixels to world space
def to_world_coords(x, y, x_pos, y_pos, yaw, world_size, scale):
    # Apply rotation
    x_rotated, y_rotated = rotate_coords(x, y, yaw)
    # Apply translation
    x_translated, y_translated = translate_coords(x_rotated, y_rotated, x_pos, y_pos, scale)
    # Perform rotation, translation and clipping
    x_world = np.clip(np.int_(x_translated), 0, world_size - 1)
    y_world = np.clip(np.int_(y_translated), 0, world_size - 1)
    # Return the result
    return x_world, y_world


# Grab image
image_coord = mpimg.imread(rock_image_path)
warped = perspect_transform(image_coord, source, destination)
path_coord, _ = color_threshold(warped)

# Calculate pixel values in rover-centric coords and distance/angle to all navigable pixels
x, y = to_rover_coords(path_coord)
radius, angle = to_polar_coords(x, y)
mean_angle = np.mean(angle)

rock_position = rock_positioning(warped)
x_rock, y_rock = to_rover_coords(rock_position)

obstacle = np.zeros_like(path_coord)
obstacle[path_coord == 0] = 1
x_obstacle, y_obstacle = to_rover_coords(obstacle)

# Do some plotting
fig = plt.figure(figsize=(12, 9))
plt.subplot(221)
plt.imshow(image_coord)
plt.subplot(222)
plt.imshow(warped)
plt.subplot(223)
plt.imshow(path_coord, cmap='gray')
plt.subplot(224)
plt.plot(x, y, '.')
plt.ylim(-160, 160)
plt.xlim(0, 160)
arrow_length = 100
x_arrow = arrow_length * np.cos(mean_angle)
y_arrow = arrow_length * np.sin(mean_angle)
plt.arrow(0, 0, x_arrow, y_arrow, color='red', zorder=2, head_width=10, width=2)
scipy.misc.imsave('OUTPUT_IMAGES/path_coord.jpg', path_coord)
plt.show()


# use the log data (from simulator) in IMG folder to plot the world map (top-down view)
data = pd.read_csv('IMG/robot_log.csv', delimiter=';', decimal='.')
images_list = data["Path"].tolist() # Create list of image pathnames
# Read in ground truth map and create a 3-channel image with it
ground_truth_map = mpimg.imread('CALIBRATION/ground_truth_map.png')
ground_truth_map_3d = np.dstack((ground_truth_map*0, ground_truth_map*255, ground_truth_map*0)).astype(np.float)

# Creating a class to be the data container
# World map is 200 x 200 grids (same size as the ground truth map: 200 x 200 pixels)
class DataBucket():
    def __init__(self):
        self.images = images_list
        self.x_pos = data["X_Position"].values
        self.y_pos = data["Y_Position"].values
        self.yaw = data["Yaw"].values
        self.count = 0 # for counting
        self.worldmap = np.zeros((200, 200, 3)).astype(np.float)
        self.ground_truth_map = ground_truth_map_3d # Ground truth worldmap

BucketData = DataBucket()

plt.imshow(ground_truth_map, cmap='gray')
plt.show()


'''
# Define a function to process stored images
def process_image(image):
    # Define source and destination points for perspective transform
    dst_size = 5
    bottom_offset = 6
    source = np.float32([[14, 140], [301, 140], [200, 96], [118, 96]])
    destination = np.float32([[image.shape[1] / 2 - dst_size, image.shape[0] - bottom_offset],
                              [image.shape[1] / 2 + dst_size, image.shape[0] - bottom_offset],
                              [image.shape[1] / 2 + dst_size, image.shape[0] - 2 * dst_size - bottom_offset],
                              [image.shape[1] / 2 - dst_size, image.shape[0] - 2 * dst_size - bottom_offset],
                              ])

    # Apply perspective transform
    warped = perspect_transform(image, source, destination)

    # Apply color threshold to identify navigable terrain/obstacles/rock samples
    path_navigable, _ = color_threshold(warped)

    # coords manipulation
    x_navigable, y_navigable = to_rover_coords(path_navigable)

    # Convert rover-centric pixel values to world coords
    x_pos = BucketData.x_pos[BucketData.count]
    y_pos = BucketData.y_pos[BucketData.count]
    yaw = BucketData.yaw[BucketData.count]
    world_size = BucketData.worldmap.shape[0]
    scale = 10

    # Navigable terrain
    x_navigable_world, y_navigable_world = to_world_coords(x_navigable, y_navigable, x_pos, y_pos, yaw, world_size, scale)

    # Rock samples
    rock_samples = rock_positioning(warped)

    if rock_samples.any():
        x_rock, y_rock = to_rover_coords(rock_samples)
        x_rock_world, y_rock_world = to_world_coords(x_rock, y_rock, x_pos, y_pos, yaw, world_size, scale)

    # Obstacles
    obstacle = np.zeros_like(path_navigable)
    obstacle[path_navigable == 0] = 1
    x_obstacle, y_obstacle = to_rover_coords(obstacle)
    x_obstacle_world, y_obstacle_world = to_world_coords(x_obstacle, y_obstacle, x_pos, y_pos, yaw, world_size, scale)

    # Update worldmap
    BucketData.worldmap[y_obstacle_world, x_obstacle_world, 0] = 255
    BucketData.worldmap[y_obstacle_world, x_obstacle_world, 1] = 0
    BucketData.worldmap[y_obstacle_world, x_obstacle_world, 2] = 0

    if rock_samples.any():
        BucketData.worldmap[y_rock_world, x_rock_world, 1] = 255
        BucketData.worldmap[y_rock_world, x_rock_world, 0] = 0
        BucketData.worldmap[y_rock_world, x_rock_world, 2] = 0

    BucketData.worldmap[y_navigable_world, x_navigable_world, 2] = 255
    BucketData.worldmap[y_navigable_world, x_navigable_world, 0] = 0
    BucketData.worldmap[y_navigable_world, x_navigable_world, 1] = 0

    # Make a mosaic image
    # First create a blank image (can be whatever shape)
    output_image = np.zeros((image.shape[0] + BucketData.worldmap.shape[0], image.shape[1] * 2, 3))
    output_image[:, :, 2] = 255
    # putting the original image in the upper left hand corner
    output_image[0:image.shape[0], 0:image.shape[1]] = image

    # Add the warped image in the upper right hand corner
    output_image[0:image.shape[0], image.shape[1]:] = warped

    # Overlay worldmap with ground truth map
    map_add = cv2.addWeighted(BucketData.worldmap, 1, BucketData.ground_truth_map, 0.5, 0)
    # Flip map overlay so y-axis points upward and add to output_image
    output_image[image.shape[0]:, 0:BucketData.worldmap.shape[1]] = np.flipud(map_add)

    # Then putting some text over the image
    cv2.putText(output_image, "Populate this image with your analysis to make a video!", (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)

    if BucketData.count < len(BucketData.images) - 1:
        BucketData.count += 1  # Keep track of the index in the DataBucket()

    return output_image
'''


def process_image(img):
    # use the DataBucket() object defined in the previous cell
    # print(data.x_pos[0], data.y_pos[0], data.yaw[0])
    warped = perspect_transform(img, source, destination)
    path_navigable, path_not_navigable = color_threshold(warped)
    rock_samples = rock_positioning(warped)

    # split rock from obstacles
    obstacles = rock_samples ^ path_not_navigable
    path_navigable = rock_samples ^ path_navigable
    vision_image = np.zeros((160, 320, 3), dtype=np.float)
    vision_image[:, :, 0] = obstacles * 255  # np.maximum(obstacles, rock_sample)
    vision_image[:, :, 1] = rock_samples * 255
    vision_image[:, :, 2] = path_navigable * 255

    output_image = np.zeros((img.shape[0] + data.worldmap.shape[0], img.shape[1] * 2, 3))

    x, y = to_rover_coords(path_navigable)
    x_rock, y_rock = to_rover_coords(rock_samples)
    x_obstacles, y_obstacles = to_rover_coords(obstacles)

    # Convert rover-centric pixel values to world coordinates
    scale = 10

    if BucketData.count >= len(BucketData.images):
        BucketData.count = len(BucketData.images) - 1
    rover_x_pos, rover_y_pos = BucketData.x_pos[BucketData.count], BucketData.y_pos[BucketData.count]
    rover_yaw = BucketData.yaw[BucketData.count]
    # print(BucketData.count, BucketData.x_pos[BucketData.count])

    # Get navigable pixel positions in world coords
    x_navigable_world, y_navigable_world = to_world_coords(x, y, rover_x_pos, rover_y_pos, rover_yaw, BucketData.worldmap.shape[0], scale)

    x_rock_world, y_rock_world = to_world_coords(x_rock, y_rock, rover_x_pos, rover_y_pos, rover_yaw, BucketData.worldmap.shape[0], scale)

    x_obstacle_world, y_obstacle_world = to_world_coords(x_obstacles, y_obstacles, rover_x_pos, rover_y_pos, rover_yaw, BucketData.worldmap.shape[0], scale)

    # Update Rover worldmap (to be displayed on right side of screen)

    BucketData.worldmap[y_obstacle_world, x_obstacle_world, 0] += 1
    BucketData.worldmap[y_rock_world, x_rock_world, 1] += 1
    BucketData.worldmap[y_navigable_world, x_navigable_world, 2] += 1

    output_image[0:img.shape[0], 0:img.shape[1]] = img
    output_image[0:img.shape[0], img.shape[1]:] = warped
    # map_add = cv2.addWeighted(BucketData.worldmap, 1, BucketData.ground_truth*255, 0.5, 0)
    # map_add = BucketData.worldmap + BucketData.ground_truth * 255 / 2
    output_image[img.shape[0]:, 0:BucketData.worldmap.shape[1]] = BucketData.worldmap * 2  # 55#ground_truth_3d * 255
    output_image[img.shape[0]:img.shape[0] + 160, img.shape[1]:] = vision_image
    cv2.putText(output_image, "Populate this image with your analysis to make a video!", (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    data.count += 1  # Keep track of the index in the Databucket()
    return output_image


# Make a video from processed image data
from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip

BucketData.count=0
BucketData.worldmap=BucketData.ground_truth_map*255*0.5

# Define pathname to save the output video
output_vedio = '../OUTPUT_IMAGES/test_mapping.mp4'
clip = ImageSequenceClip(BucketData.images, fps=60) # Note: output video will be sped up because recording rate in simulator is fps=25
new_clip = clip.fl_image(process_image) # color images!!
new_clip.write_videofile(output_vedio, audio=False)
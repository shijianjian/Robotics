import numpy as np
import cv2

from rover_state import RoverRuningState

class Helpers():
    # Identify pixels above the threshold
    # Threshold of RGB > 160 does a nice job of identifying ground pixels only
    @staticmethod
    # Identify pixels above the threshold
    # Threshold of RGB > 160 does a nice job of identifying ground pixels only
    def color_thresh(img, rgb_thresh=(160, 160, 160), above=True):
        # Create an array of zeros same xy size as img, but single channel
        color_select = np.zeros_like(img[:,:,0])
        # Require that each pixel be above all three threshold values in RGB
        # above_thresh will now contain a boolean array with "True"
        # where threshold was met
        if above:
            above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                        & (img[:,:,1] > rgb_thresh[1]) \
                        & (img[:,:,2] > rgb_thresh[2])
            # Index the array of zeros with the boolean array and set to 1
            color_select[above_thresh] = 1
            # Return the binary image
            return color_select
        else:
            below_thresh = (img[:,:,0] <= rgb_thresh[0]) \
                        & (img[:,:,1] <= rgb_thresh[1]) \
                        & (img[:,:,2] <= rgb_thresh[2])
            color_select[below_thresh] = 1
            return color_select

    # Define a function to convert to radial coords in rover space
    @staticmethod
    def to_polar_coords(x_pixel, y_pixel):
        # Convert (x_pixel, y_pixel) to (distance, angle) 
        # in polar coordinates in rover space
        # Calculate distance to each pixel
        dist = np.sqrt(x_pixel**2 + y_pixel**2)
        # Calculate angle away from vertical for each pixel
        angles = np.arctan2(y_pixel, x_pixel)
        return dist, angles

    # Define a function to map rover space pixels to world space
    @staticmethod
    def rotate_pix(xpix, ypix, yaw):
        # Convert yaw to radians
        yaw_rad = yaw * np.pi / 180
        xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                                
        ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
        # Return the result  
        return xpix_rotated, ypix_rotated

    @staticmethod
    def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
        # Apply a scaling and a translation
        xpix_translated = (xpix_rot / scale) + xpos
        ypix_translated = (ypix_rot / scale) + ypos
        # Return the result  
        return xpix_translated, ypix_translated

    # Define a function to perform a perspective transform
    @staticmethod
    def perspect_transform(img, src, dst):
            
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
        
        return warped

class Utils():

    @staticmethod
    def terrain_thresh(img):
        return Helpers.color_thresh(img, above=True)

    @staticmethod
    def obstacle_thresh(img):
        return Helpers.color_thresh(img, above=False)

    @staticmethod
    def rock_thresh(img):
        lower_bound = (120, 110, 0)
        upper_bound = (205, 180, 70)
        color_select = np.zeros_like(img[:,:,0])

        final_thresh = (np.logical_and(img[:,:,0] >= lower_bound[0], img[:,:,0] <= upper_bound[0])) \
                    &  (np.logical_and(img[:,:,1] >= lower_bound[1], img[:,:,1] <= upper_bound[1])) \
                    &  (np.logical_and(img[:,:,2] >= lower_bound[2], img[:,:,2] <= upper_bound[2]))
        color_select[final_thresh] = 1
        return color_select

    # Define a function to convert from image coords to rover coords
    @staticmethod
    def rover_coords(binary_img):
        # Identify nonzero pixels
        ypos, xpos = binary_img.nonzero()
        # Calculate pixel positions with reference to the rover position being at the 
        # center bottom of the image.  
        x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
        y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
        return x_pixel, y_pixel

    # Define a function to apply rotation and translation (and clipping)
    # Once you define the two functions above this function should work
    @staticmethod
    def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
        # Apply rotation
        xpix_rot, ypix_rot = Helpers.rotate_pix(xpix, ypix, yaw)
        # Apply translation
        xpix_tran, ypix_tran = Helpers.translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
        # Perform rotation, translation and clipping all at once
        x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
        y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
        # Return the result
        return x_pix_world, y_pix_world

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    xpos = np.array(Rover.pos[0]).astype(np.float)
    ypos = np.array(Rover.pos[1]).astype(np.float)
    yaw = np.array(Rover.yaw).astype(np.float)
    world_size = 200
    scale = 10
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    img = Rover.img
    # 1) Define source and destination points for perspective transform
    dst_size = 5 
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[img.shape[1]/2 - dst_size, img.shape[0] - bottom_offset],
                      [img.shape[1]/2 + dst_size, img.shape[0] - bottom_offset],
                      [img.shape[1]/2 + dst_size, img.shape[0] - 2*dst_size - bottom_offset], 
                      [img.shape[1]/2 - dst_size, img.shape[0] - 2*dst_size - bottom_offset],
                      ])
    # TODO: 
    # 1) Define source and destination points for perspective transform
    # 2) Apply perspective transform
    warped = Helpers.perspect_transform(img, source, destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    obstacle_threshed = Utils.obstacle_thresh(warped)
    rock_threshed = Utils.rock_thresh(warped)
    terrain_threshed = Utils.terrain_thresh(warped)
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    Rover.vision_image[:,:,0] = obstacle_threshed * 255
    Rover.vision_image[:,:,1] = rock_threshed * 255
    Rover.vision_image[:,:,2] = terrain_threshed * 255
    # 5) Convert map image pixel values to rover-centric coords
    obstacle_xpix, obstacle_ypix = Utils.rover_coords(obstacle_threshed)
    rock_xpix, rock_ypix = Utils.rover_coords(rock_threshed)
    terrain_xpix, terrain_ypix = Utils.rover_coords(terrain_threshed)
    # 6) Convert rover-centric pixel values to world coordinates
    terrain_x_world, terrain_y_world = Utils.pix_to_world(terrain_xpix, terrain_ypix, xpos, ypos, yaw, world_size, scale)
    obstacle_x_world, obstacle_y_world = Utils.pix_to_world(obstacle_xpix, obstacle_ypix, xpos, ypos, yaw, world_size, scale)
    rock_x_world, rock_y_world = Utils.pix_to_world(rock_xpix, rock_ypix, xpos, ypos, yaw, world_size, scale)
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
    Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
    Rover.worldmap[terrain_y_world, terrain_x_world, 2] += 1
    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    
    # Should not make any move during the perception stage.
    if Rover.in_searching:
        Rover.nav_dists, Rover.nav_angles = Helpers.to_polar_coords(terrain_xpix, terrain_ypix)
    else:
        dist, angle = Helpers.to_polar_coords(rock_xpix, rock_ypix)
        if not len(angle) == 0:
            Rover.nav_dists = dist
            Rover.nav_angles = angle
            Rover.brake = 0
        else:
            Rover.brake = Rover.brake_set
    return Rover
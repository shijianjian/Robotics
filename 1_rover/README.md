## Project: Search and Sample Return
**Robotics Software Engineer Nanodegree**

---

### Notebook Analysis
##### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.

Finding ground, obstacles and samples was done in a few steps:
- Finding the color range for yellow samples on the test images:
   ```python
   # Threshold for identyfying yellow samples
   sample_low_thresh = (120, 110, 0)
   sample_high_thresh = (205, 180, 70)
   ```
- Three functions were used for detecting different targets.
   ```python
    def terrain_thresh(img):
        return Helpers.color_thresh(img, above=True)

    def obstacle_thresh(img):
        return Helpers.color_thresh(img, above=False)

    def rock_thresh(img):
        lower_bound = (120, 110, 0)
        upper_bound = (205, 180, 70)
        color_select = np.zeros_like(img[:,:,0])

        final_thresh = (np.logical_and(img[:,:,0] >= lower_bound[0], img[:,:,0] <= upper_bound[0])) \
                    &  (np.logical_and(img[:,:,1] >= lower_bound[1], img[:,:,1] <= upper_bound[1])) \
                    &  (np.logical_and(img[:,:,2] >= lower_bound[2], img[:,:,2] <= upper_bound[2]))
        color_select[final_thresh] = 1
        return color_select
   ```
    Some examples can be found inside the notebook.

##### 2. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result.

The process present in the python notebook code is similar to the one that happens in the `perception_step()` described below.

---

### Autonomous Navigation and Mapping

##### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.

This is how I modified the `perception_step()` function:

- First I created the variables that hold transform source and destination values:
   ```python
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - bottom_offset],
                      [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - bottom_offset],
                      [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset],
                      [Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset],
                      ])
   ```
- I have applied a perspective transform:
   ```python
   warped = perspect_transform(Rover.img, source, destination)
   ```
- I have created thresholds for ground, obstacles and samples:
   ```python
    obstacle_threshed = Utils.obstacle_thresh(warped)
    rock_threshed = Utils.rock_thresh(warped)
    terrain_threshed = Utils.terrain_thresh(warped)
   ```
- I updated the Rover images to display in Unity:
   ```python
    Rover.vision_image[:,:,0] = obstacle_threshed * 255
    Rover.vision_image[:,:,1] = rock_threshed * 255
    Rover.vision_image[:,:,2] = terrain_threshed * 255
   ```
- Getting rover-centric coords:
   ```python
    obstacle_xpix, obstacle_ypix = Utils.rover_coords(obstacle_threshed)
    rock_xpix, rock_ypix = Utils.rover_coords(rock_threshed)
    terrain_xpix, terrain_ypix = Utils.rover_coords(terrain_threshed)
   ```
- Getting world coordinates:
   ```python
   terrain_x_world, terrain_y_world = Utils.pix_to_world(terrain_xpix, terrain_ypix, xpos, ypos, yaw, world_size, scale)
    obstacle_x_world, obstacle_y_world = Utils.pix_to_world(obstacle_xpix, obstacle_ypix, xpos, ypos, yaw, world_size, scale)
    rock_x_world, rock_y_world = Utils.pix_to_world(rock_xpix, rock_ypix, xpos, ypos, yaw, world_size, scale)
   ```
- Update worldmap **if** pitch and roll are close to 0, to increase map fidelity:
   ```python
   if Rover.pitch < Rover.max_pitch and Rover.roll < Rover.max_roll:
        Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        Rover.worldmap[terrain_y_world, terrain_x_world, 2] += 1
   ```
- And finally create polar coordinates. There are two modes created, one is for searching around for the golden rock, another is for approaching to the rock once it has been found.
   ```python
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
   ```

This is how I modified the `decision_step()`:

- There gonna be 4 modes in total. 'forward' for moving, 'turn' for turning, 'pickup' for picking up rocks and 'stop' when all rocks were picked up.
   ```python
    def decision_step(Rover): 

        if Rover.nav_angles is not None:
            print(Rover.mode)
            if Rover.mode == 'forward':
                Moves.forward(Rover)               
            if Rover.mode == 'turn':
                Moves.turn(Rover)
            if Rover.mode == 'pickup':
                Moves.pickup(Rover)
        
        if Rover.samples_collected == Rover.samples_to_find:
            Moves.stop(Rover)
            
        return Rover
   ```

- I have added two perception state to the rover, one for searching for golden rocks, one for detecting if no ways out.
    ```python
    class RoverRuningState():
    
    DEAD_LANE_THRESHOLD = 40  # Rover.vision_image[:, :, 2].shape[0]/5
    SEE_SAMPLE_THRESHOLD = 20 # If 20 pixels were seen

    @staticmethod
    def is_stucked(Rover):
        terrain = Rover.vision_image[:, :, 2].nonzero()
        terrain_m = [None, None]
        # if the rover is facing obstales
        mask = np.logical_and(
            np.array(terrain[1]) > (Rover.vision_image[:, :, 2].shape[1]/2 - RoverRuningState.DEAD_LANE_THRESHOLD),
            np.array(terrain[1]) < (Rover.vision_image[:, :, 2].shape[1]/2 + RoverRuningState.DEAD_LANE_THRESHOLD)
            )
        terrain_m[0] = terrain[0][mask]
        terrain_m[1] = terrain[1][mask]
        if len(terrain_m[1]) < 20:
            return True
        if terrain_m[0].max() < 10:
            return True
        return False

    @staticmethod
    def see_sample(Rover):
        sample = Rover.vision_image[:, :, 1].nonzero()
        if len(sample[0]) == 0:
            return False
        if sample[0].max() - sample[0].min() < RoverRuningState.SEE_SAMPLE_THRESHOLD and \
        sample[1].max() - sample[1].min() < RoverRuningState.SEE_SAMPLE_THRESHOLD:
            return True
        return False
    ```
- The most important thing is to handle stuck. With following settings, the rover will try moving forward or backward or rotating itself for getting itself out.
    ```python
    def __stucked__(Rover):
        Rover.brake = 0
        if len(Rover.nav_angles) != 0:
            if np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15) < 0:
                Rover.steer = -15
            else:
                Rover.steer = 15
        else:
            Rover.steer = 15
        if np.random.random() > 0.6:
            Rover.throttle = -Rover.throttle_set
        elif np.random.random() > 0.6:
            Rover.throttle = Rover.throttle_set
        else:
            Rover.throttle = 0
            
        time.sleep(1)
    ```
- Moving forward, if see the rock, go picking. If it get stucked, try turn around.
   ```python
   def forward(Rover):
        if RoverRuningState.see_sample(Rover) or Rover.near_sample:
            Rover.in_searching = False
            Rover.samples_located += 1
            Rover.mode = 'pickup'
        else:
            if Rover.mode == 'forward':
                # Turn some degrees if no way to go.
                if RoverRuningState.is_facing_dead_lane(Rover):
                    Rover.mode = 'turn'
                else:
                    Rover.steer = 0
                    # If stucked
                    if Rover.vel == 0 and Rover.throttle != 0:
                        Moves.__stucked__(Rover)
                    elif Rover.vel > Rover.max_vel:
                        Rover.throttle = 0
                        Rover.brake = Rover.brake_set
                    else:
                        Rover.throttle = Rover.throttle_set
                        Rover.brake = 0
                    if len(Rover.nav_angles) == 0:
                        Rover.mode = 'turn'
                    else:
                        Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
   ```
- To pick up the rock, it will stop around it then send the picking signal.
   ```python
   def pickup(Rover):
        if Rover.mode == 'pickup':
            if Rover.near_sample:
                if Rover.vel != 0:
                    Rover.steer = 0
                    Rover.throttle = 0
                    Rover.brake = Rover.brake_set
                else:
                    Rover.send_pickup = True
            else:
                if Rover.send_pickup:
                    Rover.steer = 0
                    Rover.throttle = 0
                    Rover.brake = 0
                    Rover.mode = 'forward'
                    Rover.send_pickup = False
                    Rover.in_searching = True
                else:
                    print('Approaching')
                    if Rover.vel < 0.1 and Rover.throttle != 0:
                        Moves.__stucked__(Rover)
                    else:
                        Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                        if Rover.vel > Rover.max_vel:
                            Rover.throttle = 0
                            Rover.brake = Rover.brake_set
                        else:
                            Rover.brake = 0
                            Rover.throttle = Rover.throttle_set
   ```
- Turning.
   ```python
   def turn(Rover):
        de = 15
        if Rover.mode == 'turn':
            if Rover.vel == 0 and Rover.throttle != 0:
                Moves.__stucked__(Rover)
            elif RoverRuningState.is_stucked(Rover):
                print('stuck')
                Rover.throttle = 0 
                Rover.brake = 0
                Rover.steer = de
                if np.random.random() > 0.6:
                    Rover.throttle = -Rover.throttle_set
                elif np.random.random() > 0.6:
                    Rover.steer = -de
                    Rover.throttle = Rover.throttle_set
            elif RoverRuningState.is_facing_dead_lane(Rover):
                if len(Rover.nav_angles) != 0:
                    if np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15) < 0:
                        de = -de
                print('facing')
                Rover.throttle = Rover.throttle_set
                Rover.brake = 0
                Rover.steer += de
            else:
                Rover.steer = 0
                Rover.throttle = Rover.throttle_set
                Rover.brake = 0
                Rover.mode = 'forward'
            # whatever the decision is, give some more time
            time.sleep(0.5)
   ```
- To stop moving when all the rocks get collected.
   ```python
   def stop(Rover):
        if Rover.vel > 0.2:
            Rover.throttle = 0
            Rover.brake = Rover.brake_set
            Rover.steer = 0
        elif Rover.vel < 0.2:
            Rover.mode = 'stop'
            
   ```

##### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

The robot can sometimes map around 50% - 80% of the map at a fidelity of 60% and higher while collecting all the rocks it saw. In my test, there is once it completes 95% of the map with 50% of fidelity and all the rocks have been collected. It took 1452 seconds to do. (I think it may depend on the random behavious of the stuck handler function and the initial facing orientation)

The problem with my implementation is that the rover will sometimes be trapped in one specific route, which probably should not go back to where it has been. Also, the mechanisms of turning and stuck detection is not perfect.

I can add more randomness on this build to make it easier to finish the job while I think it may not be that approprate to do for the self-driving system.

---

### Simulator settings

| Resolution | Graphics quality | FPS |
| :-- | :-- | :-- |
| 840x524 | Good | 60 |
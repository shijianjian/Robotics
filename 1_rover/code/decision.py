import numpy as np
import time

from rover_state import RoverState, RoverRuningState

class Moves():

    position_records = []

    @staticmethod
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

    @staticmethod
    def forward(Rover):
        if RoverRuningState.see_sample(Rover) or Rover.near_sample:
            Rover.in_searching = False
            Rover.samples_located += 1
            Rover.mode = 'pickup'
        else:
            if Rover.mode == 'forward':
                # Turn some degrees if no way to go.
                if RoverRuningState.is_stucked(Rover):
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

    @staticmethod
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

    @staticmethod
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
            else:
                Rover.steer = 0
                Rover.throttle = Rover.throttle_set
                Rover.brake = 0
                Rover.mode = 'forward'
            # whatever the decision is, give some more time
            time.sleep(0.5)

    @staticmethod
    def stop(Rover):
        if Rover.vel > 0.2:
            Rover.throttle = 0
            Rover.brake = Rover.brake_set
            Rover.steer = 0
        elif Rover.vel < 0.2:
            Rover.mode = 'stop'
            
    
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
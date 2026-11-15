#!/usr/bin/env python

import os
from yumi_tht_control.arm import Arm
from hexaglide.robot import Hexaglide  # CREATE HEXAGLIDE PACKAGE!!

'''
Class definitions
'''


class THTAssemblyPlan():
    '''
    THTAssemblyPlan tells which parts need to inserted into what primary holes with which orientation and in what order
    '''

    def __init__(self):
        self.tht_components = []  # list of THTComponent objects

    def order(self):
        # sort THT components by their Manhattan distance from the furthest corner of PCB from the cameras
        # modify self.tht_components list
        pass


class THTComponent():
    '''
    THTComponents have different types, container locations, presentation states etc.
    '''

    def __init__(self):
        self.component_id = None  # String like 'A', 'S84-500', 'LCP-GF40' etc.
        self.body_type = None  # String like 'cylindrical_vertical', 'cylindrical_horizontal', 'rectangular' etc.
        self.pin_config = None  # String like 'single_row', 'two_rows', 'misc' (if too many rows or pins not aligned in rows at all)
        self.pin_number = None  # Integer just for the total number of pins (e.g. 9, 24 etc.)
        self.pin_separation = None  # For single_row, 2 pin components: nominal separation between pins to reject badly deformed parts
        self.part_container = None  # THTContainer object that tells where to get this THTComponent and how the parts are presented there
        self.primary_pin_target = None  # Coordinates of the target hole for the primary pin on PCB (for the produced product)


class THTContainer():
    '''
    THTContainer is described by its location w.r.t. robot, number of remaining components, orientation state etc.
    '''

    def __init__(self):
        self.container_id = None
        self.mixed_components = None  # Boolean whether this container is supposed to have different THT component IDs in it
        self.component_ids = []  # List of Strings (which component IDs can be found here)
        self.location = None  # tbd
        self.n_remaining_components = {}  # Key: component ID, value: how many of those are left in the container
        self.presentation_state = None  # String like 'form_vertical' or 'not_fixed'
        self.orientation_state = None  # String like 'all_same_orientation' or 'various_orientations'


class THTManipulator():
    '''
    In our case YuMi or Hexaglide. THTManipulators differ by the mapping of commands and the way they are initialized.
    '''

    def __init__(self, robot_type=None):
        self.robot_type = robot_type  # String: 'yumi' or 'hexaglide'
        if self.robot_type is not None:
            if self.robot_type == 'yumi':
                self.robot = Arm(arm_group='left_arm', gripper_pub='/yumi/gripper_l_effort_cmd')
            elif self.robot_type == 'hexaglide':
                self.robot = Hexaglide(control_mode='pos')
            else:
                '''
                Raise error for invalid robot type
                '''
                pass
        else:
            '''
            Raise error for invalid robot type
            '''
            pass
        self.reference_success = self.reference_robot()

    def reference_robot(self):
        if self.robot_type == 'yumi':
            '''
            Initialize yumi arm controller
            '''
            return True
        elif self.robot_type == 'hexaglide':
            '''
            Initialize hexaglide controller
            '''
            move_down_velocity = [2.5, 2.5, 2.5, 2.5, 2.5, 2.5]
            return self.robot.doReference(move_down_velocity)

'''
Helper functions
'''

'''
This script is used to control the left arm of YuMi
'''
if __name__ == '__main__':
    '''
    General idea (can be modified or extended):
    1. Load / enter assembly plan:
	Which part needs to be inserted into what primary hole with which orientation?
		What type of part is it? (cylindrical/rectangular/special + 2-pins/DIP)
		In which container is the part located?
		Is the part standing up in the container or is it lying on its side? (actually we can't insert parts lying on the side because of the gripper…)
		For 2-pins part: what is the nominal distance between pins?
		For DIP: how many pins are there?
	In what order? (beginning from the furthest corner of the PCB from the cameras)
    Enter primary pin goal coordinates somehow (maybe based on letters on the PCB)
    2. Execute assembly plan:
	    In the assembly order detect parts and grpping points in their containers
	    From arm home position go to gripping position
    	Close the gripper
	    Move arm to cameras position depending on the part type (2-pins above tilda plane, DIP below the tilda plane)
	    Do optical referencing depending on the part type
	    Execute insertion procedure for the part and target hole (don't forget to reset pressure to 0 after a successful insertion)
	    Go to gripping position directly (first to some intermediate position above the containers)
        Repeat for the next part until assembly plan completed
    '''

    pass

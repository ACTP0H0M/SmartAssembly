from openai import robot_env
# from yumi_tht_control.arm import Arm
import numpy as np
import rospy
from std_msgs.msg import Int64, Float64MultiArray, Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from tht_control.observer import Observer

class ManipulatorEnv(robot_env.RobotGazeboEnv):
    """Superclass for all Robot environments.
    """

    def __init__(self):
        """Initializes a new Robot environment.
        """
        print("Started initializing ManipulatorEnv...")
        # Variables that we give through the constructor.

        # Internal Vars

        # self.yumi = Arm(arm_group='left_arm', gripper_pub='/yumi/gripper_l_effort_cmd')
        #self.obs = Observer(x1=550, y1=550, x2=600, y2=600, h=100, ph=10)
        #self.pressure_sub = rospy.Subscriber('/pressure', Float32, self.obs.current_pressure)
        #self.img1_sub = rospy.Subscriber("/cam_1/color/image_raw", Image, self.obs.image_callback_1)
        #self.img2_sub = rospy.Subscriber("/cam_2/color/image_raw", Image, self.obs.image_callback_2)
        #reset_controls_bool = False

        # We launch the init function of the Parent Class robot_env.RobotGazeboEnv

        super(ManipulatorEnv, self).__init__()#controllers_list=[],
                                       #  robot_name_space='yumi',
                                       #  reset_controls=reset_controls_bool)
        print("Finished initializing ManipulatorEnv")

    # Methods needed by the RobotGazeboEnv
    # ----------------------------
    def _pause_env(self):
        input("Press Enter to continue")

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        rospy.Subscriber(self.pressure_pub, )

        # TODO
        return True

    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()

    # Methods that the TrainingEnvironment will need.
    # ----------------------------
from yumi_sim import arm

yumi = Arm(arm_group='left_arm', gripper_pub='/yumi/gripper_l_effort_cmd')
print(yumi.getJointState)
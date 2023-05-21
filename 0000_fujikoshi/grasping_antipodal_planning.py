import math
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.yumi_gripper.yumi_gripper as yg
import robot_sim.end_effectors.gripper.robotiq85.robotiq85 as rtq85
base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)
# object
object_tube = cm.CollisionModel("objects/workpiece_before.stl")
object_tube.set_rgba([.9, .75, .35, .3])
object_tube.attach_to(base)
# hnd_s
gripper_s = rtq85.Robotiq85()
grasp_info_list = gpa.plan_grasps(gripper_s, object_tube,
                                  angle_between_contact_normals=math.radians(177),
                                  openning_direction='loc_y',
                                  max_samples=15, min_dist_between_sampled_contact_points=.005,
                                  contact_offset=.005)
# gpa.write_pickle_file('workpiece_before_show', grasp_info_list, './', 'robotiq85_fujikoshi.pickle')
gpa.write_pickle_file('workpiece_before_test', grasp_info_list, './', 'robotiq85_fujikoshi.pickle')
for grasp_info in grasp_info_list:

    jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    gripper_s.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
    gripper_s.gen_meshmodel(rgba=(0,1,0,0.1)).attach_to(base)
    break

base.run()
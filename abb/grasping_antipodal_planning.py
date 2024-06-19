import math
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.dh60.dh60 as dh

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)

# object
object_tube = cm.CollisionModel("objects/box.stl")
object_tube.set_rgba([.9, .75, .35, 1])
object_tube.attach_to(base)

# hnd_s
gripper_s = dh.Dh60()
grasp_info_list = gpa.plan_grasps(gripper_s, object_tube,
                                  angle_between_contact_normals=math.radians(177),
                                  openning_direction='loc_x',
                                  max_samples=10, min_dist_between_sampled_contact_points=.005,
                                  contact_offset=.005)
gpa.write_pickle_file('box', grasp_info_list, './', 'dh60_grasps.pickle')
# grasp_info_list = gpa.load_pickle_file('holder', './', 'cobg_holder_grasps.pickle')

for grasp_info in grasp_info_list:
    jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    gripper_s.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
    gripper_s.gen_meshmodel(rgba=[0,1,0,.01]).attach_to(base)

base.run()
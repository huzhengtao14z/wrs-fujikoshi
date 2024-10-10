import math
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.yumi_gripper.yumi_gripper as yg
import robot_sim.end_effectors.gripper.robotiq85.robotiq85 as rtq85
import robot_sim.end_effectors.gripper.robotiqhe.robotiqhe as rtqhe
import trimeshwraper as tw

base = wd.World(cam_pos=[1, 1, 1],w=960,
                 h=540, lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)
# object
# name = 'Tortoise_800_tex'
name = 'ratchet.stl'
# address = "3dcnnobj/"+name+".stl"

# address = "3dcnnobj/"+name+".stl"
# address = "kit_model/"+name+".obj"
address = "test_obj"
# address = "3dcnnobj/"
mesh = tw.TrimeshHu(meshpath = address, name = name )
m =mesh.outputTrimesh
object = cm.CollisionModel(m)


object.set_rgba([.9, .75, .35, 1])
# object_tube.set_scale((0.5,0.5,0.5))
object.attach_to(base)
# base.run()

# hnd_s
# gripper_s = rtq85.Robotiq85()
gripper_s = rtqhe.RobotiqHE()
grasp_info_list = gpa.plan_grasps(gripper_s, object,
                                  angle_between_contact_normals=math.radians(170),
                                  openning_direction='loc_x',
                                  rotation_interval=math.radians(30),
                                  max_samples=10, min_dist_between_sampled_contact_points=.005,
                                  contact_offset=.002)
gpa.write_pickle_file(name, grasp_info_list, '.', 'grasp/rbq85.pickle')
for grasp_info in grasp_info_list:
    jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    gripper_s.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
    gripper_s.gen_meshmodel(rgba=(1,0,0,0.1)).attach_to(base)
base.run()
import copy
import math
import numpy as np
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.dh60.dh60 as dh
import robot_sim.robots.gofa5.gofa5 as gf5
import manipulation.pick_place_planner as ppp
import motion.probabilistic.rrt_connect as rrtc
import basis.robot_math as rm

if __name__ == '__main__':
    base = wd.World(cam_pos=[4.16951, 1.8771, 1.70872], lookat_pos=[0, 0, 0.5])
    gm.gen_frame().attach_to(base)

    rbt_s = gf5.GOFA5()
    gripper_s = dh.Dh60()

    rrtc_s = rrtc.RRTConnect(rbt_s)
    ppp_s = ppp.PickPlacePlanner(rbt_s)



    rbt_s.hnd.open()
    rbt_s.gen_meshmodel().attach_to(base)
    manipulator_name = "arm"
    # start_conf = rbt_s.get_jnt_values(manipulator_name)
    start_conf = np.array([0.0439823 , -0.53023103  ,1.05243354 , 0.0143117  , 1.55351757 , 1.57079633])
    # rbt_s.fk("arm", start_conf)
    # pos, rot = rbt_s.get_gl_tcp("arm")
    # rbt_s.ik(manipulator_name, pos, rot)
    # rbt_s.fk("arm", rbt_s.ik(manipulator_name, pos, rot))
    # rbt_s.gen_meshmodel().attach_to(base)

    # base.run()
    print(start_conf)
    hand_name = "hnd"
    # object
    objcm_name = "box"
    obj = cm.CollisionModel(f"objects/{objcm_name}.stl")
    obj.set_rgba([.9, .75, .35, 1])
    obj.set_pos(np.array([.4,-.2,.015]))
    obj.set_rotmat(rm.rotmat_from_axangle(axis=[0,0,1], angle=-np.pi/4))
    obj.attach_to(base)

    obj_goal = cm.CollisionModel(f"objects/{objcm_name}.stl")
    obj_goal.set_rgba([1, 1, 1, 1])
    obj_goal.set_pos(np.array([.3,.4,.015]))
    obj_goal.set_rotmat()
    obj_goal.attach_to(base)
    # base.run()
    grasp_info_list = gpa.load_pickle_file(objcm_name, root=None, file_name='dh60_grasps.pickle')

    for grasp_info in grasp_info_list:
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        # gripper_s.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
        homomat = rm.homomat_from_posrot(obj.get_pos(), obj.get_rotmat())
        # print(obj.get_rotmat())
        # print(obj.get_pos())
        pos = rm.homomat_transform_points(homomat, hnd_pos)
        # gripper_s.grip_at_with_jcpose(pos, jaw_center_rotmat, jaw_width)
        # gripper_s.
        gripper_s.fix_to(pos=pos,rotmat=obj.get_rotmat().dot(hnd_rotmat))
        gripper_s.gen_meshmodel(rgba=[0, 1, 0, .1]).attach_to(base)
        # gm.gen_sphere(pos).attach_to(base)
        gm.gen_frame(pos, obj.get_rotmat().dot(hnd_rotmat)).attach_to(base)

        # rbt_s.gen_meshmodel().attach_to(base)
        try:
            jnts = rbt_s.ik("arm", np.asarray([jaw_center_pos + obj.get_pos()]), obj.get_rotmat().dot(hnd_rotmat))
            rbt_s.fk("arm", jnt_values = jnts)
            pos, rot = rbt_s.get_gl_tcp("arm")
            # print()
            if not rbt_s.is_collided():
                rbt_s.gen_meshmodel().attach_to(base)
            # print("here")
            #     jnts = rbt_s.ik("arm", pos+np.array([0,0,0.1]), rot)
            #     pos, rot = rbt_s.fk("arm", jnt_values=jnts)
                # rbt_s.gen_meshmodel().attach_to(base)
                break
        except:
            pass
    base.run()


    obgl_start_homomat = rm.homomat_from_posrot(obj.get_pos(), obj.get_rotmat())
    obgl_goal_homomat = rm.homomat_from_posrot(obj_goal.get_pos(), obj_goal.get_rotmat())

    conf_list, jawwidth_list, objpose_list = \
        ppp_s.gen_pick_and_place_motion(hnd_name=hand_name,
                                        objcm=obj,
                                        grasp_info_list=grasp_info_list,
                                        start_conf=np.array([0, 0, 0, 0, 0, 0]),
                                        end_conf=None,
                                        goal_homomat_list=[obgl_start_homomat, obgl_goal_homomat],
                                        approach_direction_list=[None, np.array([0, 0, -1])],
                                        approach_distance_list=[.001] * 2,
                                        depart_direction_list=[np.array([0, 0, 1]), None],
                                        depart_distance_list=[.001] * 2)
    robot_attached_list = []
    object_attached_list = []
    counter = [0]


    def update(robot_s,
               object_box,
               robot_path,
               jawwidth_path,
               obj_path,
               robot_attached_list,
               object_attached_list,
               counter,
               task):
        if counter[0] >= len(robot_path):
            counter[0] = 0
        if len(robot_attached_list) != 0:
            for robot_attached in robot_attached_list:
                robot_attached.detach()
            for object_attached in object_attached_list:
                object_attached.detach()
            robot_attached_list.clear()
            object_attached_list.clear()
        pose = robot_path[counter[0]]
        robot_s.fk(manipulator_name, pose)
        robot_s.jaw_to(hand_name, jawwidth_path[counter[0]])
        robot_meshmodel = robot_s.gen_meshmodel()
        robot_meshmodel.attach_to(base)
        robot_attached_list.append(robot_meshmodel)
        obj_pose = obj_path[counter[0]]
        objb_copy = object_box.copy()
        objb_copy.set_rgba([1, 0, 0, 1])
        objb_copy.set_homomat(obj_pose)
        objb_copy.attach_to(base)
        object_attached_list.append(objb_copy)
        counter[0] += 1
        return task.again


    taskMgr.doMethodLater(0.01, update, "update",
                          extraArgs=[rbt_s,
                                     obj,
                                     conf_list,
                                     jawwidth_list,
                                     objpose_list,
                                     robot_attached_list,
                                     object_attached_list,
                                     counter],
                          appendTask=True)
    base.run()
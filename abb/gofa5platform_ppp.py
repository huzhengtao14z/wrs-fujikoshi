import copy
import math
import numpy as np
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.dh60.dh60 as dh
import robot_sim.robots.GOFA5.GOFA5 as gf5
import manipulation.pick_place_planner as ppp
import motion.probabilistic.rrt_connect as rrtc
import basis.robot_math as rm

if __name__ == '__main__':
    base = wd.World(cam_pos=[4.16951, 1.8771, 1.70872], lookat_pos=[0, 0, 0.5])
    gm.gen_frame().attach_to(base)

    rbt_s = gf5.GOFA5()
    rbt_s.hnd.open()
    # rbt_s.gen_meshmodel().attach_to(base)
    manipulator_name = "arm"
    # start_conf = rbt_s.get_jnt_values(manipulator_name)
    start_conf = np.array([0.0439823 , -0.53023103  ,1.05243354 , 0.0143117  , 1.55351757 , 1.57079633])
    print(start_conf)
    hand_name = "hnd"
    # object
    objcm_name = "box"
    obj = cm.CollisionModel(f"objects/{objcm_name}.stl")
    obj.set_rgba([.9, .75, .35, 1])
    obj.set_pos(np.array([.5,-.2,0.25]))
    obj.set_rotmat()
    obj.attach_to(base)

    # object_goal
    obj_goal = cm.CollisionModel(f"objects/{objcm_name}.stl")
    obj_goal.set_rgba([1, 1, 1, 1])
    obj_goal.set_pos(np.array([.6,.4,0.25]))
    obj_goal.set_rotmat()
    obj_goal.attach_to(base)

    gripper_s = dh.Dh60()
    grasp_info_list = gpa.load_pickle_file(objcm_name, root=None, file_name='dh60_grasps.pickle')

    # for grasp_info in grasp_info_list:
    #     jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    #     gripper_s.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
    #     pos = gripper_s.pos + np.array([.4, -.2, -.015])
    #     gripper_s.fix_to(pos=pos,rotmat=hnd_rotmat)
    #     gripper_s.gen_meshmodel(rgba=[0, 1, 0, .1]).attach_to(base)
    #
    # for grasp_info in grasp_info_list:
    #     jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    #     gripper_s.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
    #     pos = gripper_s.pos + np.array(([.3, .4, -.015]))
    #     gripper_s.fix_to(pos=pos,rotmat=hnd_rotmat)
    #     gripper_s.gen_meshmodel(rgba=[0, 1, 0, .1]).attach_to(base)

    start_pos = obj_goal.get_pos()
    start_rotmat = obj_goal.get_rotmat()
    start_homo = rm.homomat_from_posrot(start_pos, start_rotmat)

    # jnts_list=[]
    # for grasp_info in grasp_info_list:
    #     jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    #     jnts = rbt_s.ik(component_name="arm", tgt_pos=rm.homomat_transform_points(start_homo,jaw_center_pos),
    #        tgt_rotmat=start_rotmat.dot(jaw_center_rotmat))
    #     if jnts is not None:
    #         jnts_list.append(jnts)
    #
    # rrtc_planner = rrtc.RRTConnect(rbt_s)
    # print("num_list_jnts", len(jnts_list))
    # for jnts in jnts_list:
    #     # print(jnts)
    #     rbt_s.fk(component_name='arm', jnt_values=jnts)
    #     pos, rot = rbt_s.get_gl_tcp("arm")
    #     pos = pos +[0,0,0.1]
    #     jnts_goal = rbt_s.ik("arm", pos, rot)
    #
    #     try:
    #         path = rrtc_planner.plan(component_name="arm",
    #                                  start_conf=np.array([0.0439823 , -0.53023103  ,1.05243354 , 0.0143117  , 1.55351757 , 1.57079633]),
    #                                  goal_conf=jnts_goal,
    #                                  ext_dist=0.05,
    #                                  max_time=300)
    #
    #         # rbt_s.fk("arm", jnts)
    #         # rbt_s.gen_meshmodel(rgba=[0, 1, 0, .1]).attach_to(base)
    #         if path is not None:
    #             print(path)
    #             for item in path:
    #                 rbt_s.fk("arm", item)
    #                 rbt_s.gen_meshmodel(rgba=[0, 1, 0, .1]).attach_to(base)
    #             break
    #     except Exception as e:
    #         pass
    #
    #
    # base.run()

    rrtc_planner = rrtc.RRTConnect(rbt_s)
    goal_pos =obj_goal.get_pos()
    goal_rotmat = obj.get_rotmat()
    # goal_jnt_values = rbt_s.ik(tgt_pos=goal_pos, tgt_rotmat=goal_rotmat)
    # rbt_s.fk(component_name="arm", jnt_values=goal_jnt_values)
    # rbt_s.gen_meshmodel().attach_to(base)
    rrtc_s = rrtc.RRTConnect(rbt_s)
    ppp_s = ppp.PickPlacePlanner(rbt_s)



    obgl_start_homomat = rm.homomat_from_posrot(start_pos, start_rotmat)
    obgl_goal_homomat = rm.homomat_from_posrot(goal_pos, goal_rotmat)

    conf_list, jawwidth_list, objpose_list = \
        ppp_s.gen_pick_and_place_motion(hnd_name=hand_name,
                                        objcm=obj,
                                        grasp_info_list=grasp_info_list,
                                        start_conf=start_conf,
                                        end_conf=None,
                                        goal_homomat_list=[obgl_start_homomat, obgl_goal_homomat],
                                        approach_direction_list=[None, np.array([0, 0, -1])],
                                        approach_distance_list=[.01] * 2,
                                        depart_direction_list=[np.array([0, 0, 1]), None],
                                        depart_distance_list=[.01] * 2)
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
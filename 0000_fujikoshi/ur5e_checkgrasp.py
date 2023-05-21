if __name__ == '__main__':
    import math
    import numpy as np
    import basis.robot_math as rm
    import robot_sim.robots.cobotta.cobotta as cbt
    # import robot_sim.robots.ur5e_conveyorbelt.ur5e_conveyorbet as ur5e
    import robot_sim.robots.ur5e_machinetool.ur5e_machinetool as ur5e
    import robot_con.cobotta.cobotta_x as cbtx
    import motion.probabilistic.rrt_connect as rrtc
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    import modeling.collision_model as cm
    import grasping.planning.antipodal as gpa
    import manipulation.pick_place_planner as ppp
    import robot_sim.end_effectors.gripper.robotiq85.robotiq85 as rbt85
    import motion.probabilistic.rrt_connect as rrtc
    # import robot_sim.manipulators.machinetool.machinetool_gripper as machine
    base = wd.World(cam_pos=[1, 1, .5], lookat_pos=[0, 0, .2])
    # gm.gen_frame(length=.5, thickness=.05,).attach_to(base)
    original_grasp_info_list = gpa.load_pickle_file('workpiece_before', './', 'robotiq85_fujikoshi.pickle')
    # jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    obj = cm.CollisionModel("objects/holder.stl")
    obj.set_pos(pos=np.array([0, 0, 1]))
    # obj.attach_to(base)
    table = cm.CollisionModel("objects/MTbase2.stl")
    table.set_pos(pos = np.array([0.7,0.2,-0.82]))
    table.attach_to(base)

    workpiece_before = cm.CollisionModel("objects/workpiece_before.stl")
    workpiece_after = cm.CollisionModel("objects/workpiece_after.stl")

    workpiece_before.set_pos(pos=np.array([0.4, 0.1, 0.0]))
    # workpiece_before.show_localframe()
    # workpiece_before.set_rpy(np.pi,0,0)
    # homo_workpiece_before = workpiece_before.get_homomat()
    workpiece_before.attach_to(base)




    manipulator_name = "arm"
    component_name = "arm"
    robot_s = ur5e.UR5EConveyorBelt()
    robot_s.door_to(1)
    start_conf = robot_s.get_jnt_values(component_name=component_name)
    # start_tcp = robot_s.fk( component_name="arm" , jnt_values=start_conf)
    start_tcp_pos, start_tcp_rot = robot_s.get_gl_tcp(manipulator_name = manipulator_name)


    # robot_meshmodel = robot_s.gen_meshmodel(is_machine=True, is_robot=False)
    robot_s.jaw_to(0.085)

    # workpiece_before.set_pos(pos=robot_s.machine.jaw_center_pos+ np.array([-0.06,0,0]))
    # workpiece_before.set_rotmat(rotmat=np.dot(robot_s.machine.jaw_center_rot, rm.rotmat_from_axangle(np.array([0,1,0]), np.pi/2)))

    homo_workpiece_before=workpiece_before.get_homomat()
    rotmat_workpiece_before = workpiece_before.get_rotmat()
    # workpiece_after.attach_to(base)
    # base.run()
    # #-----------------
    grpr = rbt85.Robotiq85(enable_cc=True)

    for id, item in enumerate(original_grasp_info_list):
        if id == 0 :
            jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = item
            grpr.grip_at_with_jcpose(rm.homomat_transform_points(homo_workpiece_before, jaw_center_pos), np.dot(rotmat_workpiece_before,jaw_center_rotmat), jaw_width)
            if grpr.is_collided(obstacle_list=[table]):
                grpr.gen_meshmodel(rgba=(1,0,0,0.1)).attach_to(base)
            else:
                grpr.gen_meshmodel().attach_to(base)
        if id%5 ==0:
            jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = item
            grpr.grip_at_with_jcpose(rm.homomat_transform_points(homo_workpiece_before, jaw_center_pos), np.dot(rotmat_workpiece_before,jaw_center_rotmat), jaw_width)
            if grpr.is_collided(obstacle_list=[table]):
                grpr.gen_meshmodel(rgba=(1,0,0,0.1)).attach_to(base)
            else:
                grpr.gen_meshmodel(rgba=(0,1,0,0.1)).attach_to(base)

    base.run()
    # #--------------------------
    goalpos_workpiece_before = workpiece_before.get_pos()
    goalhomo_workpiece_before = workpiece_before.get_homomat()



    robot_s.fk(component_name, start_conf)
    ppp_s = ppp.PickPlacePlanner(robot_s)
    # start_conf = robot_s.get_jnt_values(manipulator_name)
    hand_name = "hnd"
    obstacle_list = []
    conf_list, jawwidth_list, objpose_list = \
        ppp_s.gen_pick_and_place_motion(hnd_name=hand_name,
                                        objcm=workpiece_before,
                                        grasp_info_list=original_grasp_info_list,
                                        start_conf=start_conf,
                                        end_conf=start_conf,
                                        obstacle_list=obstacle_list,
                                        goal_homomat_list=[homo_workpiece_before, goalhomo_workpiece_before ],
                                        approach_direction_list=[None, np.array([1, 0, 0])],
                                        approach_distance_list=[.20] * 2,
                                        depart_direction_list=[np.array([0, 0, 1]), np.array([-1, 0, 0])],
                                        depart_distance_list=[.20] * 2)
    #===========================================
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # rrtc_planner = rrtc.RRTConnect(robot_s)
    # for item in original_grasp_info_list[200:]:
    #     # jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    #     jnts = robot_s.ik(component_name=component_name,
    #                tgt_pos=rm.homomat_transform_points(homo_workpiece_before, item[1]),
    #                tgt_rotmat=item[2])
    #     if jnts is not None:
    #         path = rrtc_planner.plan(component_name="arm",
    #                                  start_conf=start_conf,
    #                                  goal_conf=jnts,
    #                                  ext_dist=0.005,
    #                                  max_time=300)
    #         if path is not None:
    #             robot_s.fk(component_name, jnts)
    #             robot_s.jaw_to(hnd_name='hnd', jawwidth=item[0])
    #             robot_s.gen_meshmodel(rgba=(0, 1, 0, 1)).attach_to(base)
    #             robot_path = path
    #             break
    #+++++++++++++++++++++++++++++++++
    # base.run()
    # jnts_values = robot_s.ik(component_name, )
    print(jawwidth_list)

    robot_attached_list = []
    object_attached_list = []
    counter = [0]
    machine_attached_list = []

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
            # for machine_attached in machine_attached_list:
            #     machine_attached.detach()
            robot_attached_list.clear()
            object_attached_list.clear()
            # machine_attached_list.clear()
        pose = robot_path[counter[0]]
        robot_s.fk(manipulator_name, pose)
        robot_s.jaw_to(jawwidth=jawwidth_path[counter[0]])
        # robot_s.door_to(door_list[counter[0]])
        # robot_s.chunck_to(chunck_list[counter[0]])
        robot_meshmodel = robot_s.gen_meshmodel(is_machine=True)
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


    object_holder = workpiece_before

    # jawwidth_list = [0.005 for i in conf_list]
    # objpose_list = [np.eye(4) for i in conf_list]

    # jawwidth_list = [0.005 for i in range(40)]
    # conf_list = [path[0] for i in range(40)]
    # objpose_list = [np.eye(4) for i in range(40)]
    # door_list = np.linspace(0, 1, 20)
    # door_list2 = np.linspace(1, 1, 20)
    # door_list = np.concatenate((door_list, door_list2), axis = 0)
    # chunck_list = np.linspace(0, 0, 20)
    # chunck_list2 = np.linspace(0, 0.08, 20)
    # chunck_list = np.concatenate((chunck_list, chunck_list2), axis=0)

    taskMgr.doMethodLater(0.01, update, "update",
                          extraArgs=[robot_s,
                                     object_holder,
                                     conf_list,
                                     jawwidth_list,
                                     objpose_list,
                                     robot_attached_list,
                                     object_attached_list,
                                     counter],
                          appendTask=True)
    base.run()
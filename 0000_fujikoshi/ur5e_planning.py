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
    # import robot_sim.manipulators.machinetool.machinetool_gripper as machine
    base = wd.World(cam_pos=[1, 1, .5], lookat_pos=[0, 0, .2])
    gm.gen_frame(length=.5, thickness=.05,).attach_to(base)

    obj = cm.CollisionModel("objects/holder.stl")
    obj.set_pos(pos = np.array([0,0,1]))
    obj.attach_to(base)

    workpiece_before = cm.CollisionModel("objects/workpiece_before.stl")
    workpiece_after = cm.CollisionModel("objects/workpiece_after.stl")
    workpiece_after.set_pos(pos=np.array([0, 0.5, 1.5]))
    workpiece_after.attach_to(base)
    workpiece_before.set_pos(pos=np.array([0, 0, 1.5]))
    workpiece_before.attach_to(base)
    # self.base_stand.jnts[1]['loc_pos'] = np.array([.9, -1.5, -0.06])
    # self.base_stand.jnts[2]['loc_pos'] = np.array([0, 1.23, 0])
    # self.base_stand.jnts[3]['loc_pos'] = np.array([0, 0, 0])
    # self.base_stand.jnts[4]['loc_pos'] = np.array([-.9, .27, 0.06])
    # gm.gen_sphere(np.array([.9, -1.5, -0.06]), radius=0.08).attach_to(base)
    # gm.gen_sphere(np.array([0, 1.23, 0]), radius=0.08).attach_to(base)
    manipulator_name = "arm"
    component_name = "arm"
    gm.gen_sphere(np.array([0, 0, 0]), radius=0.08).attach_to(base)
    gm.gen_sphere(np.array([-.9, .27, 0.06]), radius=0.08).attach_to(base)


    # machine = machine.RobotiqHE()
    # machine.gen_meshmodel().attach_to(base)

    robot_s = ur5e.UR5EConveyorBelt()
    start_conf = robot_s.get_jnt_values(component_name=component_name)
    # start_tcp = robot_s.fk( component_name="arm" , jnt_values=start_conf)
    start_tcp_pos, start_tcp_rot = robot_s.get_gl_tcp(manipulator_name = manipulator_name)
    gm.gen_sphere(start_tcp_pos, radius=0.02).attach_to(base)
    robot_meshmodel = robot_s.gen_meshmodel()
    robot_meshmodel.attach_to(base)
    robot_s.jaw_to(0.05)
    # robot_s.fk("arm", pose)
    start_tcp_pos, start_tcp_rot = robot_s.get_gl_tcp(manipulator_name = manipulator_name)
    gm.gen_sphere(start_tcp_pos, radius=0.02).attach_to(base)
    robot_meshmodel = robot_s.gen_meshmodel()
    robot_meshmodel.attach_to(base)

    goal_tcp_pos = start_tcp_pos + np.array([-0.2,-0.8,-0.50])
    rotmat = rm.rotmat_from_axangle(np.array([0,1,0]), angle = 30*np.pi/180)
    goal_tcp_rot = np.dot(rotmat, start_tcp_rot)
    rotmat = rm.rotmat_from_axangle(np.array([1, 0, 0]), angle=10 * np.pi / 180)
    goal_tcp_rot = np.dot(rotmat, goal_tcp_rot)
    goal_jnt_values = robot_s.ik(tgt_pos=goal_tcp_pos, tgt_rotmat=goal_tcp_rot)
    print("start_radians", start_conf)
    # print("start_tcp", start_tcp)
    # tgt_pos = np.array([.25, .2, .15])
    # tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi * 2/ 3)
    # jnt_values = robot_s.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
    jnt_values = goal_jnt_values
    rrtc_planner = rrtc.RRTConnect(robot_s)
    path = rrtc_planner.plan(component_name="arm",
                             start_conf=start_conf,
                             goal_conf=jnt_values,
                             ext_dist=0.1,
                             max_time=300)
    # for pose in path:
    #     print(pose)
    #     robot_s.fk("arm", pose)
    #     robot_meshmodel = robot_s.gen_meshmodel()
    #     robot_meshmodel.attach_to(base)

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
        robot_s.jaw_to(jawwidth_path[counter[0]])
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


    object_holder = obj
    conf_list = path
    jawwidth_list = [0.005 for i in path]
    objpose_list = [np.eye(4) for i in path]
    taskMgr.doMethodLater(0.1, update, "update",
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
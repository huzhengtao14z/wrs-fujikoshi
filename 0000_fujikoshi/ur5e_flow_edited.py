from direct.task.TaskManagerGlobal import taskMgr

if __name__ == '__main__':
    # .. Needed modulues and Lib
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
    import motion.probabilistic.rrt_connect as rrtc
    # import robot_sim.manipulators.machinetool.machinetool_gripper as machine
    original_grasp_info_list = gpa.load_pickle_file('workpiece_before', './', 'robotiq85_fujikoshi.pickle') # .. Needed for pick and place parameters
    # ..Create base
    base = wd.World(cam_pos=[1, 1, .5], lookat_pos=[0, 0, .2])
    # ..Create table (1)
    table_1 = cm.CollisionModel("objects/MTbase2.stl")
    table_1.set_pos(pos=np.array([0.7, 0.2, -0.82]))
    table_1.attach_to(base)
    # ..Create table (2)
    table_2 = cm.CollisionModel("objects/MTbase2.stl")
    table_2.set_pos(pos=np.array([0.7, 1.6, -0.82]))
    table_2.attach_to(base)
    #base.run()
    # ..Create Workpiece(1) for table(1), set its position, show its local frame and calculate its homogenous matrix
    workpiece_before = cm.CollisionModel("objects/workpiece_before.stl")
    workpiece_before.set_pos(pos=np.array([0.4, 0.1, 0.0]))
    workpiece_before.show_localframe()
    homo_workpiece_before = workpiece_before.get_homomat()
    workpiece_before.attach_to(base)
    # ..Create Workpiece(2) for table(1), set its position, show its local frame and calculate its homogenous matrix
    workpiece_21_before = cm.CollisionModel("objects/workpiece_before.stl")
    workpiece_21_before.set_pos(pos=np.array([0.4, 0.4, 0.0]))
    workpiece_21_before.set_rgba(np.array([255 / 255, 255 / 255, 0, 1]))
    workpiece_21_before.show_localframe()
    homo_workpiece_21_before = workpiece_before.get_homomat()
    workpiece_21_before.attach_to(base)
    # ..Create Workpiece(1) for table(2), set its position, show its local frame and calculate its homogenous matrix
    workpiece_12_before = cm.CollisionModel("objects/workpiece_before.stl")
    workpiece_12_before.set_pos(pos=np.array([0.4, 1.5, 0.0]))
    workpiece_12_before.show_localframe()
    homo_workpiece_12_before = workpiece_before.get_homomat()
    workpiece_12_before.attach_to(base)
    # ..Create Workpiece(2) for table(2), set its position, show its local frame and calculate its homogenous matrix
    workpiece_22_before = cm.CollisionModel("objects/workpiece_before.stl")
    workpiece_22_before.set_pos(pos=np.array([0.4, 1.8, 0.0]))
    workpiece_22_before.set_rgba(np.array([255 / 255, 255 / 255, 0, 1]))
    workpiece_22_before.show_localframe()
    homo_workpiece_22_before = workpiece_before.get_homomat()
    workpiece_22_before.attach_to(base)
    # .. Create machine (1) and set its configurations ( type, door condition, jaw dimension, start config, tcp pos and rot)
    manipulator_name = "arm"
    component_name = "arm"
    robot_s = ur5e.UR5EConveyorBelt()
    robot_s.door_to(1)
    start_conf = robot_s.get_jnt_values(component_name=component_name)
    start_tcp_pos, start_tcp_rot = robot_s.get_gl_tcp(manipulator_name=manipulator_name)
    robot_s.jaw_to(0.085)
    # .. If is_robot is false, only the machine will be included.
    robot_meshmodel = robot_s.gen_meshmodel(is_machine=True, is_robot=False)
    robot_meshmodel.attach_to(base)
    # robot_s.fk(component_name, start_conf)
    # ppp_s = ppp.PickPlacePlanner(robot_s)
    # base.run()
    # .. Create machine (2) and set its configurations ( type, door condition, jaw dimension, start config, tcp pos and rot)
    manipulator_2_name = "arm"
    component_2_name = "arm"
    robot_s_2 = ur5e.UR5EConveyorBelt(pos=np.array([0, 1.6, 0]), enable_cc=True)
    robot_s_2.door_to(1)
    start_conf_2 = robot_s_2.get_jnt_values(component_name=component_2_name)
    start_tcp_2_pos, start_tcp_2_rot = robot_s_2.get_gl_tcp(manipulator_name=manipulator_2_name)
    robot_s_2.jaw_to(0.085)
    # .. If is_robot is false, only the machine will be included.
    robot_2_meshmodel = robot_s_2.gen_meshmodel(is_machine=False, is_robot=True)
    robot_2_meshmodel.attach_to(base)
    # robot_s_2.fk(component_2_name, start_conf_2)
    ppp_s_2 = ppp.PickPlacePlanner(robot_s_2)
    # .. To configure the pick and place hand manipulator line 57
    hand_name = "hnd"
    base.run()
    # .. Link the workpiece position and orientation with the machine jaw center position and orientation
    workpiece_before.set_pos(pos=robot_s.machine.jaw_center_pos + np.array([-0.06, 0, 0]))
    workpiece_before.set_rotmat(rotmat=np.dot(robot_s.machine.jaw_center_rot, rm.rotmat_from_axangle(np.array([0, 1, 0]), np.pi / 2)))

    # .. Set the goal position for the robot to be the workpiece position and orientation
    goal_pos_workpiece_before = workpiece_before.get_pos()
    goal_homo_workpiece_before = workpiece_before.get_homomat()

    # .. Config obstacle list between start pos and end pos if any to check collision
    obstacle_list = []
    # .. Configure the motion planning using pick and place
    conf_list, jaw_width_list, obj_pose_list = \
        ppp_s.gen_pick_and_place_motion(hnd_name=hand_name,
                                        objcm=workpiece_before,
                                        grasp_info_list=original_grasp_info_list,
                                        start_conf=start_conf,
                                        end_conf=start_conf,
                                        obstacle_list=obstacle_list,
                                        goal_homomat_list=[homo_workpiece_before, goal_homo_workpiece_before],
                                        approach_direction_list=[None, np.array([1, 0, 0])],
                                        approach_distance_list=[.20] * 2,
                                        depart_direction_list=[np.array([0, 0, 1]), np.array([-1, 0, 0])],
                                        depart_distance_list=[.20] * 2)
    # .. Config the machine door and chunk parameters before motion
    door_list_before = np.linspace(0, 1, 50)
    door_list2 = np.linspace(1, 1, 50)
    door_list_before = np.concatenate((door_list_before, door_list2), axis=0)
    chunk_list_before = np.linspace(0, 0, 50)
    chunk_list2 = np.linspace(0, 0.08, 50)                           # .. Note that the second number represent the chunk head openning distance
    chunk_list_before = np.concatenate((chunk_list_before, chunk_list2), axis=0)

    # .. Create a function that calculate the path concatenated before

    def path_concatenated(door_list, chunk_list, jaw_width_list, obj_pose_list, conf_list):
        rbt_door = [door_list[-1] for i in range(len(conf_list))]
        rbt_chunk = [chunk_list[-1] for i in range(len(conf_list))]

        conf_list_door = [conf_list[0] for i in range(len(door_list))]
        jaw_width_list_door = [jaw_width_list[0] for i in range(len(door_list))]
        obj_pose_list_door = [obj_pose_list[0] for i in range(len(door_list))]

        door_list = np.concatenate((door_list, rbt_door), axis=0)
        chunk_list = np.concatenate((chunk_list, rbt_chunk), axis=0)
        conf_list = np.concatenate((conf_list_door, conf_list), axis=0)
        jaw_width_list = np.concatenate((jaw_width_list_door, jaw_width_list), axis=0)
        obj_pose_list = np.concatenate((obj_pose_list_door, obj_pose_list), axis=0)
        return door_list, chunk_list, conf_list, jaw_width_list, obj_pose_list

    # .. function usage
    door_list, chunk_list, conf_list, jaw_width_list, obj_pose_list = path_concatenated(door_list_before,
                                                                                       chunk_list_before,
                                                                                       jaw_width_list, obj_pose_list,
                                                                                       conf_list)
    # .. Config the machine door and chunk parameters after motion
    door_list_after = np.linspace(1, -0.6, 50)
    door_list2 = np.linspace(-0.6, -0.6, 50)
    door_list_after = np.concatenate((door_list_after, door_list2), axis=0)
    chunk_list_after = np.linspace(0, 0, 50)
    chunk_list2 = np.linspace(0, 0, 50)
    chunk_list_after = np.concatenate((chunk_list_after, chunk_list2), axis=0)
    # .. Config the path concatenated after motion
    door_list = np.concatenate((door_list, door_list_after), axis = 0)
    chunk_list = np.concatenate((chunk_list, chunk_list_after), axis=0)

    conf_list_door = [conf_list[-1] for i in range(len(door_list_after))]
    jaw_width_list_door = [jaw_width_list[-1] for i in range(len(door_list_after))]
    obj_pose_list_door = [obj_pose_list[-1] for i in range(len(door_list_after))]

    conf_list = np.concatenate((conf_list, conf_list_door), axis = 0)
    jaw_width_list = np.concatenate((jaw_width_list, jaw_width_list_door), axis=0)
    obj_pose_list = np.concatenate((obj_pose_list, obj_pose_list_door), axis=0)
    # .. Config required variables for upgrade
    robot_attached_list = []
    object_attached_list = []
    counter = [0]
    machine_attached_list = []


    # .. Create a function that update the path of the robot


    def update(robot_s,
               object_box,
               robot_path,
               jaw_width_path,
               obj_path,
               robot_attached_list,
               object_attached_list,
               machine_attached_list,
               door_list,
               chunk_list,
               counter,
               task):
        if counter[0] >= len(robot_path):
            counter[0] = 0
        if len(robot_attached_list) != 0:
            for robot_attached in robot_attached_list:
                robot_attached.detach()
            for object_attached in object_attached_list:
                object_attached.detach()
            for machine_attached in machine_attached_list:
                machine_attached.detach()
            robot_attached_list.clear()
            object_attached_list.clear()
            machine_attached_list.clear()
        pose = robot_path[counter[0]]
        robot_s.fk(manipulator_name, pose)
        robot_s.jaw_to(jawwidth=jaw_width_path[counter[0]])
        robot_s.door_to(door_list[counter[0]])
        robot_s.chunck_to(chunk_list[counter[0]])
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

    taskMgr.doMethodLater(0.01, update, "update",
                          extraArgs=[robot_s,
                                     object_holder,
                                     conf_list,
                                     jaw_width_list,
                                     obj_pose_list,
                                     robot_attached_list,
                                     object_attached_list,
                                     machine_attached_list,
                                     door_list,
                                     chunk_list,
                                     counter],
                          appendTask=True)
    base.run()
    ...
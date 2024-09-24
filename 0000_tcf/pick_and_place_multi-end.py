import math
import visualization.panda.world as wd
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.grippers.yumi_gripper.yumi_gripper as yg
import robot_sim.end_effectors.grippers.robotiqhe.robotiqhe as hnde
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode
import numpy as np
import basis.robot_math as rm
import modeling.geometric_model as gm
import robot_sim.robots.ur3_dual.ur3_dual as ur3d
import robot_sim.robots.ur3e_dual.ur3e_dual as ur3ed
import robot_sim.robots.sda5f.sda5f as sda5
import motion.probabilistic.rrt_connect as rrtc
import manipulation.pick_place_planner as ppp
import os
import pickle
import basis.data_adapter as da
import robot_sim.end_effectors.grippers.robotiqhe.robotiqhe as rtqhe
import slope
import Sptpolygoninfo as sinfo


def show_ikfeasible_poses(obj_rotmat, obj_pos):
    hndfa = rtqhe.RobotiqHE(enable_cc=True)
    obj_fixture = object.copy()
    obj_fixture.set_rotmat(obj_rotmat)
    obj_fixture.set_pos(obj_pos)
    obj_fixture.attach_to(base)
    for i, item in enumerate(grasp_info_list):
        jaw_width, gl_jaw_center_pos, gl_jaw_center_rotmat, hnd_pos, hnd_rotmat = item
        hnd_rotmat = obj_rotmat.dot(gl_jaw_center_rotmat)
        hnd_pos = obj_rotmat.dot(gl_jaw_center_pos) + obj_pos

        hndfa.fix_to(pos=hnd_pos,
                     rotmat=hnd_rotmat, jawwidth=.04)

        # hndfa.gen_meshmodel(rgba=(1, 0, 0, 0.05)).attach_to(base)
        jnt_values = robot.ik(component_name=component_name,
                              tgt_pos=hnd_pos,
                              tgt_rotmat=hnd_rotmat,
                              max_niter=500,
                              toggle_debug=False,
                              seed_jnt_values=None)
        if jnt_values is not None:
            robot.fk(component_name=component_name, jnt_values=jnt_values)

            if hndfa.is_mesh_collided(slopeforcd_high, toggle_debug=False):
                robot.gen_meshmodel(toggle_tcpcs=False, rgba=(1, 0, 0, 0.05)).attach_to(base)
                # pass
            else:
                robot.gen_meshmodel(toggle_tcpcs=False, rgba=(0, 1, 0, 0.5)).attach_to(base)
                print(f"IK Done!, feasible grasp ID {i}")


if __name__ == '__main__':
    # world
    base = wd.World(cam_pos=[2.01557, 0.637317, 1.88133], w=960,
                    h=540, lookat_pos=[0, 0, 1.1])
    gm.gen_frame().attach_to(base)
    this_dir, this_filename = os.path.split(__file__)

    object = cm.CollisionModel("./objects/test_long.stl")
    object.set_scale((.001,.001,.001))
    object.set_rgba([.5, .7, .3, 1])
    gm.gen_frame().attach_to(object)

    # object_goal = object.copy()
    # object_goal_pos = np.array([0.800, -.300, 0.900])
    # object_goal_rotmat = np.eye(3)
    # object_goal_homomat = rm.homomat_from_posrot(object_goal_pos, object_goal_rotmat)
    # object_goal.set_pos(object_goal_pos)
    # object_goal.set_rotmat(object_goal_rotmat)
    # object_goal.attach_to(base)

    fixture = cm.CollisionModel("./objects/tc100.stl")
    fixture.set_scale((.001, .001, .001))
    fixture.set_rgba([.8, .6, .3, 0.5])
    fixture_start = fixture.copy()
    fixture_start_pos = np.array([0.800, -.150, 0.780])
    fixture_start_rotmat = np.eye(3)
    fixture_start_homomat = rm.homomat_from_posrot(fixture_start_pos, fixture_start_rotmat)
    fixture_start_rotmat = fixture_start_homomat[:3,:3]
    fixture_start.set_pos(fixture_start_pos)
    fixture_start.set_rotmat(fixture_start_rotmat)
    fixture_start.attach_to(base)

    slopename = "tc71.stl"
    slope_high = slope.Slope(z=-0.005, placement="ping", size=sinfo.Sloperate[slopename], show=False)
    slopeforcd_high = slope_high.getSlope()
    for i, item in enumerate(slopeforcd_high):

        item.set_homomat(fixture_start_homomat.dot(item.get_homomat()))
        slopeforcd_high[i] = item
    # slopeforcd_high[0].attach_to(base)
    # slopeforcd_high[1].attach_to(base)
    # slopeforcd_high[2].attach_to(base)

    robot = ur3ed.UR3EDual()
    component_name = 'rgt_arm'
    # robot_meshmodel = robot.gen_meshmodel()
    # robot_meshmodel.attach_to(base)

    address = this_dir + "/PlacementData"
    objname = "test_long"
    with open(address + "/" + objname + "/" + "placementrotmat.pickle", "rb") as f:
        RotMat = pickle.load(f)
    with open(address + "/" + objname + "/" + "stablecklist.pickle", "rb") as f:
        stablecklist = pickle.load(f)
    with open(address + "/" + objname + "/" + "placementcom.pickle", "rb") as f:
        comlistall = pickle.load(f)

    RotMatnozero = [RotMat[i] for i in range(len(stablecklist)) if stablecklist[i] is True]
    RotMatnozeroID = [i for i in range(len(stablecklist)) if stablecklist[i] is True]
    comlist = [comlistall[i] for i in range(len(stablecklist)) if stablecklist[i] is True]

    object_fixture = object.copy()
    object_fixture_pos = fixture_start_pos + np.array([0,0,0.00])
    object_fixture_rotmat = np.eye(3)
    object_fixture_homomat = rm.homomat_from_posrot(object_fixture_pos, object_fixture_rotmat).dot(da.pdmat4_to_npmat4(RotMatnozero[15]))
    object_fixture.set_homomat(object_fixture_homomat)
    # object_fixture.attach_to(base)

    object_fixture_after = object.copy()
    object_fixture_after_pos = fixture_start_pos + np.array([0, 0, 0.05])
    object_fixture_after_rotmat = np.eye(3)
    object_fixture_after_homomat = rm.homomat_from_posrot(object_fixture_after_pos, object_fixture_after_rotmat).dot(
        da.pdmat4_to_npmat4(RotMatnozero[12]))
    object_fixture_after.set_homomat(object_fixture_after_homomat)
    # object_fixture_after.attach_to(base)
    # base.run()
    # object_fixture.attach_to(base)

    # object_goal = object_fixture

    object_start = object.copy()
    object_start_pos = np.array([0.900, -.350, 0.800])
    object_start_rotmat = rm.rotmat_from_axangle((1,0,0), np.radians(-90)).dot(rm.rotmat_from_axangle((0,0,1),np.radians(180)))
    object_start_homomat = rm.homomat_from_posrot(object_start_pos, object_start_rotmat)
    object_start.set_pos(object_start_pos)
    object_start.set_rotmat(object_start_rotmat)
    # object_start.attach_to(base)

    grasp_info_list = gpa.load_pickle_file('test_long', './', 'PlacementData/rtqhe.pickle')

    # show_ikfeasible_poses(object_fixture_homomat[:3,:3], object_fixture_pos)
    # show_ikfeasible_poses(object_fixture_after_homomat[:3, :3], object_fixture_after_pos)
    # base.run()

    object_fixture_corner = object.copy()
    object_fixture_corner_pos = fixture_start_pos
    object_fixture_corner_rotmat = np.eye(3)
    object_fixture_corner_homomat = rm.homomat_from_posrot(object_fixture_corner_pos, object_fixture_corner_rotmat).dot(
        da.pdmat4_to_npmat4(RotMatnozero[1]))
    object_fixture_corner.set_homomat(object_fixture_corner_homomat)

    object_goal = object.copy()
    object_goal_pos = np.array([0.550, -.050, 0.850])
    object_goal_rotmat = rm.rotmat_from_axangle((0,0,1),np.radians(90)).dot(rm.rotmat_from_axangle((0,1,0),np.radians(-90)))
    object_goal_homomat = rm.homomat_from_posrot(object_goal_pos, object_goal_rotmat)
    object_goal.set_pos(object_goal_pos)
    object_goal.set_rotmat(object_goal_rotmat)

    rrtc = rrtc.RRTConnect(robot)
    ppp = ppp.PickPlacePlanner(robot)

    hand_name = "rgt_arm"
    start_conf = robot.get_jnt_values(component_name)
    # conf_list, jawwidth_list, objpose_list = \
    #     ppp.gen_pick_and_place_motion(hnd_name=hand_name,
    #                                   objcm=object,
    #                                   grasp_info_list=grasp_info_list,
    #                                   start_conf=start_conf,
    #                                   end_conf=start_conf,
    #                                   goal_homomat_list=[object_start_homomat, object_fixture_homomat],
    #                                   approach_direction_list=[None, np.array([0, 0, -1])],
    #                                   approach_distance_list=[.05] * 2,
    #                                   depart_direction_list=[np.array([0, 0, 1]), None],
    #                                   depart_distance_list=[.05] * 2)
    conf_list, jawwidth_list, objpose_list = \
        ppp.gen_pick_and_place_motion(hnd_name=hand_name,
                                      objcm=object,
                                      grasp_info_list=grasp_info_list,
                                      start_conf=start_conf,
                                      end_conf=start_conf,
                                      goal_homomat_list=[object_fixture_homomat, object_fixture_after_homomat],
                                      approach_direction_list=[None, np.array([0, 0, -1])],
                                      approach_distance_list=[.05] * 2,
                                      obstacle_list=slopeforcd_high,
                                      depart_direction_list=[np.array([0, 0, 1]), None],
                                      depart_distance_list=[.1] * 2)




    # conf_list, jawwidth_list, objpose_list = \
    #     ppp.gen_pick_and_place_motion(hnd_name=hand_name,
    #                                   objcm=object,
    #                                   grasp_info_list=grasp_info_list,
    #                                   start_conf=start_conf,
    #                                   end_conf=start_conf,
    #                                   goal_homomat_list=[object_fixture_corner_homomat, object_goal_homomat],
    #                                   approach_direction_list=[None, np.array([0, 0, -1])],
    #                                   approach_distance_list=[.1] * 2,
    #                                   depart_direction_list=[np.array([0, 0, 1]), None],
    #                                   obstacle_list=slopeforcd_high,
    #                                   depart_distance_list=[.1, .1])
    middle_conf_list = [conf_list[i] for i in range(len(conf_list)) if jawwidth_list[i] < 0.04]

    robot.fk(hand_name, middle_conf_list[0])
    robot.jaw_to(hand_name, 0.03)
    robot_meshmodel = robot.gen_meshmodel()
    robot_meshmodel.attach_to(base)
    objb_copy = object.copy()
    objb_copy.set_rgba([0, 191 / 255, 1, 1])
    objb_copy.set_homomat(objpose_list[0])
    objb_copy.attach_to(base)

    robot.fk(hand_name, middle_conf_list[-1])
    robot.jaw_to(hand_name, 0.03)
    robot_meshmodel = robot.gen_meshmodel()
    robot_meshmodel.attach_to(base)
    objb_copy = object.copy()
    objb_copy.set_rgba([0, 191 / 255, 1, 1])
    objb_copy.set_homomat(objpose_list[-1])
    objb_copy.attach_to(base)

    for i in range(0,len(conf_list), 1):
        robot.fk(hand_name, conf_list[i])
        if jawwidth_list[i] <0.04:
            robot.jaw_to(hand_name, jawwidth_list[i])
            robot_meshmodel = robot.gen_meshmodel()
            tcp = robot.get_gl_tcp(hand_name)
            gm.gen_sphere(tcp[0], radius=0.005).attach_to(base)
            # robot_meshmodel.attach_to(base)
            # robot_attached_list.append(robot_meshmodel)
            obj_pose = objpose_list[i]
            objb_copy = object.copy()
            objb_copy.set_rgba([0, 191 / 255, 1, 0.08])
            objb_copy.set_homomat(obj_pose)
            objb_copy.attach_to(base)
        # object_attached_list.append(objb_copy)
    base.run()

    # print(len(conf_list), conf_list)
    # if conf_list is None:
    #     exit(-1)
    robot_attached_list = []
    object_attached_list = []
    counter = [0]
    textNode = [None, None, None]


    # base.run()
    def update(robot,
               object_box,
               robot_path,
               jawwidth_path,
               obj_path,
               robot_attached_list,
               object_attached_list,
               counter, textNode,
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
        robot.fk(hand_name, pose)
        robot.jaw_to(hand_name, jawwidth_path[counter[0]])
        robot_meshmodel = robot.gen_meshmodel()
        robot_meshmodel.attach_to(base)
        robot_attached_list.append(robot_meshmodel)
        obj_pose = obj_path[counter[0]]
        objb_copy = object_box.copy()
        objb_copy.set_rgba([0, 191 / 255, 1, 1])
        objb_copy.set_homomat(obj_pose)
        objb_copy.attach_to(base)
        object_attached_list.append(objb_copy)
        counter[0] += 1

        if textNode[0] is not None:
            textNode[0].detachNode()
            textNode[1].detachNode()
            textNode[2].detachNode()
        cam_pos = base.cam.getPos()
        textNode[0] = OnscreenText(
            text=str(cam_pos[0])[0:5],
            fg=(1, 0, 0, 1),
            pos=(1.0, 0.8),
            align=TextNode.ALeft)
        textNode[1] = OnscreenText(
            text=str(cam_pos[1])[0:5],
            fg=(0, 1, 0, 1),
            pos=(1.3, 0.8),
            align=TextNode.ALeft)
        textNode[2] = OnscreenText(
            text=str(cam_pos[2])[0:5],
            fg=(0, 0, 1, 1),
            pos=(1.6, 0.8),
            align=TextNode.ALeft)

        return task.again


    taskMgr.doMethodLater(0.01, update, "update",
                          extraArgs=[robot,
                                     object,
                                     conf_list,
                                     jawwidth_list,
                                     objpose_list,
                                     robot_attached_list,
                                     object_attached_list,
                                     counter, textNode],
                          appendTask=True)
    # base.run()
    #
    #
    # def update(textNode, task):
    #     if textNode[0] is not None:
    #         textNode[0].detachNode()
    #         textNode[1].detachNode()
    #         textNode[2].detachNode()
    #     cam_pos = base.cam.getPos()
    #     textNode[0] = OnscreenText(
    #         text=str(cam_pos[0])[0:5],
    #         fg=(1, 0, 0, 1),
    #         pos=(1.0, 0.8),
    #         align=TextNode.ALeft)
    #     textNode[1] = OnscreenText(
    #         text=str(cam_pos[1])[0:5],
    #         fg=(0, 1, 0, 1),
    #         pos=(1.3, 0.8),
    #         align=TextNode.ALeft)
    #     textNode[2] = OnscreenText(
    #         text=str(cam_pos[2])[0:5],
    #         fg=(0, 0, 1, 1),
    #         pos=(1.6, 0.8),
    #         align=TextNode.ALeft)
    #     return task.again

    cam_view_text = OnscreenText(
        text="Camera View: ",
        fg=(0, 0, 0, 1),
        pos=(1.15, 0.9),
        align=TextNode.ALeft)
    # testNode = [None, None, None]
    # taskMgr.add(update, "addobject", extraArgs=[testNode], appendTask=True)
    base.run()
import math
import numpy as np
import basis.robot_math as rm
import robot_sim.robots.GOFA12.GOFA12 as cbt
import robot_con.cobotta.cobotta_x as cbtx
import motion.probabilistic.rrt_connect as rrtc
import visualization.panda.world as wd
import modeling.geometric_model as gm
import robot_sim.end_effectors.handgripper.finger.leftfinger.leftfinger as lfgr
import time


if __name__ == '__main__':

    start = time.time()
    base = wd.World(cam_pos=[1, 1, .5], lookat_pos=[0, 0, .2])
    gm.gen_frame().attach_to(base)

    finger = lfgr.Leftfinger(enable_cc=True)
    goal_jnt_list = np.array([10*np.pi/180, 10*np.pi/180, 10*np.pi/180])
    start_jnt_list = np.array([0, 0, 0])
    finger_mesh = finger.gen_meshmodel()
    finger_mesh.attach_to(base)
    finger_mesh.show_cdprimit()

    finger.fk(goal_jnt_list)
    finger_mesh = finger.gen_meshmodel()
    finger_mesh.attach_to(base)
    finger_mesh.show_cdprimit()
    base.run()
    print(finger.get_jnt_values())

    rrtc_planner = rrtc.RRTConnect(finger)
    path = rrtc_planner.plan(component_name="arm",
                             start_conf=start_jnt_list,
                             goal_conf=goal_jnt_list,
                             ext_dist=0.01,
                             max_time=300)
    for pose in path[1:-1]:
        finger.fk(pose)
        finger_meshmodel = finger.gen_meshmodel()
        finger_meshmodel.attach_to(base)

    base.run()
    robot_s = cbt.GOFA12()
    # robot_s.gen_meshmodel().attach_to(base)
    start_conf = robot_s.get_jnt_values(component_name='arm')
    print("start_radians", start_conf)
    # base.run()
    # tgt_pos = np.array([.25, .2, .15])
    # tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi * 2/ 3)
    # jnt_values = robot_s.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
    jnt_values = start_conf + np.array([0.5, 0.5, 0.5, 0.5,0.5,0.5])
    robot_s.fk(component_name="arm", jnt_values=jnt_values)

    robot_s.gen_meshmodel(rgba=(1, 0, 0, 1)).attach_to(base)


    tgt_pos = robot_s.get_gl_tcp("arm")[0]
    tgt_rotmat = robot_s.get_gl_tcp("arm")[1]

    goal_pos = tgt_pos + np.array([0, 0, -0.5])
    goal_rotmat = tgt_rotmat
    goal_jnt_values = robot_s.ik(tgt_pos=goal_pos, tgt_rotmat=goal_rotmat)
    robot_s.fk(component_name="arm", jnt_values=goal_jnt_values)
    robot_s.gen_meshmodel(rgba=(0, 1, 0, 1)).attach_to(base)

    # base.run()

    rrtc_planner = rrtc.RRTConnect(robot_s)
    path = rrtc_planner.plan(component_name="arm",
                             start_conf=jnt_values,
                             goal_conf=goal_jnt_values,
                             ext_dist=5,
                             max_time=300)
    for pose in path[1:-1]:
        robot_s.fk("arm", pose)
        robot_meshmodel = robot_s.gen_meshmodel()
        robot_meshmodel.attach_to(base)
        end = time.time()
        print('程序运行时间为: %s Seconds' % (end - start))
    base.run()

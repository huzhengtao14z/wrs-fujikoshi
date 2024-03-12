import os
import math
import numpy as np
import basis.robot_math as rm
import modeling.geometric_model as gm
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.manipulator_interface as mi


class MZ12HArm(mi.ManipulatorInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), homeconf=np.zeros(6), name='MZ12H', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        self.jlc = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=homeconf, name=name)
        # six joints, n_jnts = 6+2 (tgt ranges from 1-6), nlinks = 6+1
        self.jlc.jnts[1]['loc_pos'] = np.array([0, 0, 0.231])
        # self.jlc.jnts[1]['motion_rng'] = [-2.97, 2.97]
        self.jlc.jnts[1]['loc_motionax'] = np.array([0, 0, 1])
        self.jlc.jnts[2]['loc_pos'] = np.array([0.150, 0.1455, 0.219])
        self.jlc.jnts[2]['loc_motionax'] = np.array([0, 1, 0])
        # self.jlc.jnts[2]['motion_rng'] = [-2.79, 1.57]
        self.jlc.jnts[3]['loc_pos'] = np.array([0, 0, 0.610])
        self.jlc.jnts[3]['loc_motionax'] = np.array([0, 1, 0])
        # self.jlc.jnts[3]['motion_rng'] = [-2.57, 3.67]
        self.jlc.jnts[4]['loc_pos'] = np.array([0.24742, -0.1455, 0.140])
        self.jlc.jnts[4]['loc_motionax'] = np.array([0, 0, 1])
        # self.jlc.jnts[4]['motion_rng'] = [-3.32, 3.32]
        self.jlc.jnts[5]['loc_pos'] = np.array([0.43258, 0, 0])
        self.jlc.jnts[5]['loc_motionax'] = np.array([0, 1, 0])
        # self.jlc.jnts[5]['motion_rng'] = [-2.44, 2.44]
        self.jlc.jnts[6]['loc_pos'] = np.array([0.0742, 0, 0])
        self.jlc.jnts[6]['loc_motionax'] = np.array([0, 0, 1])
        # self.jlc.jnts[6]['motion_rng'] = [-6.28, 6.28]
        self.jlc.jnts[7]['loc_pos'] = np.array([0, 0, 0.01])
        self.jlc.jnts[7]['loc_rotmat'] = rm.rotmat_from_euler(0, math.pi / 2.0, 0)
        # links
        self.jlc.lnks[0]['name'] = "base"
        self.jlc.lnks[0]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[0]['mass'] = 16.71922
        self.jlc.lnks[0]['com'] = np.array([-.04698, 0, .12304])
        self.jlc.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "base.dae")
        self.jlc.lnks[0]['rgba'] = [.7,.7,.7, 1]
        self.jlc.lnks[1]['name'] = "j1"
        self.jlc.lnks[1]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[1]['com'] = np.array([.06136, .01887, .14487])
        self.jlc.lnks[1]['mass'] = 20.91729
        self.jlc.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "Joi1.dae")
        self.jlc.lnks[1]['rgba'] = [.7,.7,.7, 1]
        self.jlc.lnks[2]['name'] = "j2"
        self.jlc.lnks[2]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[2]['com'] = np.array([.1439, -0.01169, .27606])
        self.jlc.lnks[2]['mass'] = 7.26565
        self.jlc.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "Joi2.dae")
        self.jlc.lnks[2]['rgba'] = [.7,.7,.7, 1]
        self.jlc.lnks[3]['name'] = "j3"
        self.jlc.lnks[3]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[3]['com'] = np.array([0.05283, .02056, .07094])
        self.jlc.lnks[3]['mass'] = 13.49445
        self.jlc.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "Joi3.dae")
        self.jlc.lnks[3]['rgba'] = [.7,.7,.7, 1]
        self.jlc.lnks[4]['name'] = "j4"
        self.jlc.lnks[4]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[4]['com'] = np.array([-0.23593, -0.00034, 0.00017])
        self.jlc.lnks[4]['mass'] = 7.60882
        self.jlc.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "Joi4.dae")
        self.jlc.lnks[4]['rgba'] = [.7,.7,.7, 1.0]
        self.jlc.lnks[5]['name'] = "j5"
        self.jlc.lnks[5]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[5]['com'] = np.array([0.03019, -0.00642, 0.00003])
        self.jlc.lnks[5]['mass'] = 0.56095
        self.jlc.lnks[5]['mesh_file'] = os.path.join(this_dir, "meshes", "J5.dae")
        self.jlc.lnks[5]['rgba'] = [.7, .7, .7, 1]
        self.jlc.lnks[6]['name'] = "j6"
        self.jlc.lnks[6]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[6]['com'] = np.array([.00741, -.00008, 0.00008])
        self.jlc.lnks[6]['mass'] = 0.04394
        self.jlc.lnks[6]['mesh_file'] = os.path.join(this_dir, "meshes", "J6.dae")
        self.jlc.lnks[6]['rgba'] = [.7,.7,.7, 1.0]
        self.jlc.reinitialize()
        # collision detection
        if enable_cc:
            self.enable_cc()

    def enable_cc(self):
        super().enable_cc()
        self.cc.add_cdlnks(self.jlc, [0, 1, 2, 3, 4, 5, 6])
        activelist = [self.jlc.lnks[0],
                      self.jlc.lnks[1],
                      self.jlc.lnks[2],
                      self.jlc.lnks[3],
                      self.jlc.lnks[4],
                      self.jlc.lnks[5],
                      self.jlc.lnks[6]]
        self.cc.set_active_cdlnks(activelist)
        fromlist = [self.jlc.lnks[0],
                    self.jlc.lnks[1]]
        intolist = [self.jlc.lnks[3],
                    self.jlc.lnks[5],
                    self.jlc.lnks[6]]
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.jlc.lnks[2]]
        intolist = [self.jlc.lnks[4],
                    self.jlc.lnks[5],
                    self.jlc.lnks[6]]
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.jlc.lnks[3]]
        intolist = [self.jlc.lnks[6]]
        self.cc.set_cdpair(fromlist, intolist)


if __name__ == '__main__':
    import time
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, .3])
    gm.gen_frame().attach_to(base)

    manipulator_instance = MZ12HArm(enable_cc=True)

    manipulator_meshmodel = manipulator_instance.gen_meshmodel()
    manipulator_meshmodel.attach_to(base)
    base.run()
    # manipulator_meshmodel.show_cdprimit()
    # manipulator_instance.gen_stickmodel(toggle_jntscs=True).attach_to(base)
    # tic = time.time()
    # print(manipulator_instance.is_collided())
    # toc = time.time()
    # print(toc - tic)

    # base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0,0,0])
    # gm.GeometricModel("./meshes/base.dae").attach_to(base)
    # gm.gen_frame().attach_to(base)
    base.run()
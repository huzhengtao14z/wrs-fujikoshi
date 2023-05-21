import os
import numpy as np
import modeling.model_collection as mc
import robot_sim._kinematics.jlchain as jl
import basis.robot_math as rm
import robot_sim.end_effectors.gripper.gripper_interface as gp
import modeling.geometric_model as gm
import modeling.collision_model as cm


class MachineTool(gp.GripperInterface):

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 coupling_offset_pos=np.zeros(3),
                 coupling_offset_rotmat=np.eye(3),
                 cdmesh_type='box',
                 name='robotiqhe',
                 enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        this_dir, this_filename = os.path.split(__file__)
        self.coupling.jnts[1]['loc_pos'] = coupling_offset_pos
        self.coupling.jnts[1]['loc_rotmat'] = coupling_offset_rotmat
        self.coupling.lnks[0]['collision_model'] = cm.gen_stick(self.coupling.jnts[0]['loc_pos'],
                                                                self.coupling.jnts[1]['loc_pos'],
                                                                thickness=.07, rgba=[.2, .2, .2, 1],
                                                                sections=24)
        self.coupling.reinitialize()
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        # - highfinger
        self.fingerhigh = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(1), name='base_lft_finger')
        self.fingerhigh.jnts[1]['loc_pos'] = np.array([-0.3, -0.035, 1.2])
        self.fingerhigh.jnts[1]['type'] = 'prismatic'
        self.fingerhigh.jnts[1]['motion_rng'] = [-1, 1]
        self.fingerhigh.jnts[1]['loc_motionax'] = np.array([0, 0, -1])

        self.fingerhigh.lnks[0]['name'] = "base"
        self.fingerhigh.lnks[0]['loc_pos'] = np.zeros(3)
        self.fingerhigh.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "MTbase_rot.stl")
        self.fingerhigh.lnks[0]['rgba'] = [.2, .2, .2, 1]

        self.fingerhigh.lnks[1]['name'] = "finger1"
        self.fingerhigh.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "MTfinger.stl")
        self.fingerhigh.lnks[1]['rgba'] = [.5, .5, .5, 1]

        # - base
        self.base = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(4),
                                     name='base')
        self.base.lnks[0]['name'] = "base1"
        self.base.lnks[0]['loc_pos'] = np.zeros(3)
        self.base.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "MTbase1.stl")
        self.base.lnks[0]['rgba'] = [.2, .2, .2, 1]

        self.base.jnts[1]['loc_pos'] = np.array([0, 0, 0])
        self.base.jnts[1]['type'] = 'prismatic'
        self.base.jnts[1]['motion_rng'] = [0, .025]
        self.base.jnts[1]['loc_motionax'] = np.array([0, 0, -1])

        self.base.lnks[1]['name'] = "base2"
        self.base.lnks[1]['loc_pos'] = np.zeros(3)
        self.base.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "MTbase2.stl")
        self.base.lnks[1]['rgba'] = [.2, .2, .2, 1]

        self.base.jnts[2]['loc_pos'] = np.array([0, 0, 0])
        self.base.jnts[2]['type'] = 'prismatic'
        self.base.jnts[2]['motion_rng'] = [0, .025]
        self.base.jnts[2]['loc_motionax'] = np.array([0, 0, -1])

        self.base.lnks[2]['name'] = "base3"
        self.base.lnks[2]['loc_pos'] = np.zeros(3)
        self.base.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "MTbase3.stl")
        self.base.lnks[2]['rgba'] = [.2, .2, .2, 1]

        self.base.jnts[3]['loc_pos'] = np.array([0, 0, 0])
        self.base.jnts[3]['type'] = 'prismatic'
        self.base.jnts[3]['motion_rng'] = [0, .025]
        self.base.jnts[3]['loc_motionax'] = np.array([0, 0, -1])

        self.base.lnks[3]['name'] = "base4"
        self.base.lnks[3]['loc_pos'] = np.zeros(3)
        self.base.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "MTbase4.stl")
        self.base.lnks[3]['rgba'] = [.2, .2, .2, 1]


        # - lowleftfinger
        self.lowleftfinger = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(1), name='lowleftfinger')
        self.lowleftfinger.jnts[1]['loc_pos'] = np.array([-0.3, -0.035, 1.2])
        self.lowleftfinger.jnts[1]['loc_rotmat'] = rm.rotmat_from_axangle(np.array([1,0,0]), 120*np.pi/180)
        self.lowleftfinger.jnts[1]['type'] = 'prismatic'
        self.lowleftfinger.jnts[1]['loc_motionax'] = np.array([0, 0, -1])
        self.lowleftfinger.lnks[1]['name'] = "lowleftfinger"
        self.lowleftfinger.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "MTfinger.stl")
        # self.lowleftfinger.lnks[1]['collision_model'] = cm.CollisionModel(self.lowleftfinger.lnks[1]['mesh_file'],
        #                 cdprimit_type="cylinder",
        #                   cdmesh_type=cdmesh_type)
        self.lowleftfinger.lnks[1]['rgba'] = [.5, .5, .5, 1]
        # - lowrigtfinger
        self.lowrightfinger = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(1), name='lowrigtfinger')
        self.lowrightfinger.jnts[1]['loc_pos'] = np.array([-0.3, -0.035, 1.2])
        self.lowrightfinger.jnts[1]['loc_rotmat'] = rm.rotmat_from_axangle(np.array([1, 0, 0]), 240 * np.pi / 180)
        self.lowrightfinger.jnts[1]['type'] = 'prismatic'
        self.lowrightfinger.jnts[1]['loc_motionax'] = np.array([0, 0, -1])
        self.lowrightfinger.lnks[1]['name'] = "lowrigtfinger"
        self.lowrightfinger.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "MTfinger.stl")
        self.lowrightfinger.lnks[1]['rgba'] = [.5, .5, .5, 1]
        # - door
        self.door = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(1), name='door')
        self.door.jnts[1]['loc_pos'] = np.array([0, .0, 0])
        self.door.jnts[1]['type'] = 'prismatic'
        self.door.jnts[1]['loc_motionax'] = np.array([-1, 0, 0])
        self.door.lnks[1]['name'] = "door"
        self.door.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "MTdoor.stl")
        self.door.lnks[1]['rgba'] = [.5, .5, .5, 1]
        # jaw center
        jaw_homo = rm.homomat_from_posrot(pos, rotmat)
        self.jaw_center_pos = rm.homomat_transform_points(jaw_homo, self.lowleftfinger.jnts[1]['loc_pos'])
        self.jaw_center_rot = rotmat
        # reinitialize
        self.fingerhigh.reinitialize()
        self.lowleftfinger.reinitialize()
        self.lowrightfinger.reinitialize()
        self.door.reinitialize()
        self.base.reinitialize()
        # collision detection
        self.all_cdelements = []
        self.enable_cc(toggle_cdprimit=enable_cc)

    def enable_cc(self, toggle_cdprimit):
        if toggle_cdprimit:
            super().enable_cc()
            # cdprimit
            self.cc.add_cdlnks(self.fingerhigh, [0, 1])
            self.cc.add_cdlnks(self.lowleftfinger, [1])
            self.cc.add_cdlnks(self.lowrightfinger, [1])
            self.cc.add_cdlnks(self.door, [1])
            self.cc.add_cdlnks(self.base, [0,1,2,3])
            activelist = [self.fingerhigh.lnks[0],
                          self.fingerhigh.lnks[1],
                          self.lowleftfinger.lnks[1],
                          self.lowrightfinger.lnks[1],
                          self.door.lnks[1],
                          self.base.lnks[0],
                          self.base.lnks[1],
                          self.base.lnks[2],
                          self.base.lnks[3]]
            self.cc.set_active_cdlnks(activelist)
            self.all_cdelements = self.cc.all_cdelements
        # cdmesh
        for cdelement in self.all_cdelements:
            cdmesh = cdelement['collision_model'].copy()
            self.cdmesh_collection.add_cm(cdmesh)

    def fix_to(self, pos, rotmat, jawwidth=None):
        self.pos = pos
        self.rotmat = rotmat
        if jawwidth is not None:
            side_jawwidth = (.05 - jawwidth) / 2.0
            if 0 <= side_jawwidth <= .025:
                self.fingerhigh.jnts[1]['motion_val'] = side_jawwidth;
                self.lowleftfinger.jnts[1]['motion_val'] = self.fingerhigh.jnts[1]['motion_val']
            else:
                raise ValueError("The angle parameter is out of range!")
        self.coupling.fix_to(self.pos, self.rotmat)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        self.fingerhigh.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.lowleftfinger.fix_to(cpl_end_pos, cpl_end_rotmat)

    def fk(self, motion_val):
        """
        lft_outer is the only active joint, all others mimic this one
        :param: angle, radian
        """
        if self.fingerhigh.jnts[1]['motion_rng'][0] <= motion_val <= self.fingerhigh.jnts[1]['motion_rng'][1]:
            self.fingerhigh.jnts[1]['motion_val'] = motion_val
            self.lowleftfinger.jnts[1]['motion_val'] = self.fingerhigh.jnts[1]['motion_val']
            self.lowrightfinger.jnts[1]['motion_val'] = self.fingerhigh.jnts[1]['motion_val']
            self.fingerhigh.fk()
            self.lowleftfinger.fk()
            self.lowrightfinger.fk()
        else:
            raise ValueError("The motion_val parameter is out of range!")

    def fk_door(self, motion_val):
        """
        lft_outer is the only active joint, all others mimic this one
        :param: angle, radian
        """
        if self.door.jnts[1]['motion_rng'][0] <= motion_val <= self.door.jnts[1]['motion_rng'][1]:
            self.door.jnts[1]['motion_val'] = motion_val
            # self.door.jnts[1]['motion_val'] = self.door.jnts[1]['motion_val']
            self.door.fk()
            # self.lowleftfinger.fk()
        else:
            raise ValueError("The motion_val parameter is out of range!")

    def jaw_to(self, jaw_width):
        if jaw_width > 1:
            raise ValueError("The jawwidth parameter is out of range!")
        self.fk(motion_val=(0.05 - jaw_width) / 2.0)

    def door_to(self, door_width):
        if door_width > 1:
            raise ValueError("The jawwidth parameter is out of range!")
        self.fk_door(motion_val=(0.05 - door_width) / 2.0)

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='robotiqe_stick_model'):
        stickmodel = mc.ModelCollection(name=name)
        self.coupling.gen_stickmodel(toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        self.fingerhigh.gen_stickmodel(toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.lowleftfinger.gen_stickmodel(toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.lowrightfinger.gen_stickmodel(toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.door.gen_stickmodel(toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.base.gen_stickmodel(toggle_tcpcs=False,
                                 toggle_jntscs=toggle_jntscs,
                                 toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        if toggle_tcpcs:
            jaw_center_gl_pos = self.rotmat.dot(self.jaw_center_pos) + self.pos
            jaw_center_gl_rotmat = self.rotmat.dot(self.jaw_center_rotmat)
            gm.gen_dashstick(spos=self.pos,
                             epos=jaw_center_gl_pos,
                             thickness=.0062,
                             rgba=[.5, 0, 1, 1],
                             type="round").attach_to(stickmodel)
            gm.gen_mycframe(pos=jaw_center_gl_pos, rotmat=jaw_center_gl_rotmat).attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='robotiqe_mesh_model'):
        meshmodel = mc.ModelCollection(name=name)
        self.coupling.gen_meshmodel(toggle_tcpcs=False,
                                    toggle_jntscs=toggle_jntscs,
                                    rgba=rgba).attach_to(meshmodel)
        self.fingerhigh.gen_meshmodel(toggle_tcpcs=False,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        self.lowleftfinger.gen_meshmodel(toggle_tcpcs=False,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        self.lowrightfinger.gen_meshmodel(toggle_tcpcs=False,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        self.door.gen_meshmodel(toggle_tcpcs=False,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        self.base.gen_meshmodel(toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs,
                                rgba=rgba).attach_to(meshmodel)
        if toggle_tcpcs:
            jaw_center_gl_pos = self.rotmat.dot(self.jaw_center_pos) + self.pos
            jaw_center_gl_rotmat = self.rotmat.dot(self.jaw_center_rotmat)
            gm.gen_dashstick(spos=self.pos,
                             epos=jaw_center_gl_pos,
                             thickness=.0062,
                             rgba=[.5, 0, 1, 1],
                             type="round").attach_to(meshmodel)
            gm.gen_mycframe(pos=jaw_center_gl_pos, rotmat=jaw_center_gl_rotmat).attach_to(meshmodel)
        return meshmodel


if __name__ == '__main__':
    import visualization.panda.world as wd
    import math

    base = wd.World(cam_pos=[5, 5, 5], lookat_pos=[0, 0, 0])
    gm.gen_frame(length=1.5).attach_to(base)
    # for angle in np.linspace(0, .85, 8):
    #     grpr = Robotiq85()
    #     grpr.fk(angle)
    #     grpr.gen_meshmodel().attach_to(base)
    grpr = RobotiqHE(coupling_offset_pos=np.array([0, 0, 0]),
                     coupling_offset_rotmat=rm.rotmat_from_axangle([1, 0, 0], 0), enable_cc=True)
    grpr.jaw_to(0)
    grpr.door_to(0.9)
    grpr.gen_meshmodel().attach_to(base)
    # grpr.gen_stickmodel(togglejntscs=False).attach_to(base)
    # grpr.fix_to(pos=np.array([0, .3, .2]), rotmat=rm.rotmat_from_axangle([1, 0, 0], .05))
    grpr.gen_meshmodel().attach_to(base)
    # grpr.show_cdmesh()
    # grpr.show_cdprimit()
    base.run()

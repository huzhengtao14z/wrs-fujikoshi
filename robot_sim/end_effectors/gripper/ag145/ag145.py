import os
import math
import numpy as np

import basis.robot_math as rm
import modeling.geometric_model as gm
import modeling.model_collection as mc
import robot_sim._kinematics.jlchain as jl
import robot_sim.end_effectors.gripper.gripper_interface as gp


class Ag145(gp.GripperInterface):
    """
    author: kiyokawa, revised by weiwei
    date: 2020212
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), cdmesh_type='box', name='ag145', enable_cc=True):

        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        this_dir, this_filename = os.path.split(__file__)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        # - lft_outer
        self.lft_outer = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(3), name='lft_outer')
        self.lft_outer.jnts[0]['loc_pos'] = np.array([0, -.0, .0])
        self.lft_outer.jnts[0]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, 0)
        self.lft_outer.jnts[1]['loc_pos'] = np.array([0, -.03675, .09334])
        self.lft_outer.jnts[1]['motion_rng'] = [0, 54.04/180*math.pi]
        self.lft_outer.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(49.73/180*math.pi+math.pi/2, 0, 0)
        self.lft_outer.jnts[1]['loc_motionax'] = np.array([-1, 0, 0])
        self.lft_outer.jnts[2]['loc_pos'] = np.array([0, 0.0865, 0])  # passive
        self.lft_outer.jnts[2]['loc_motionax'] = np.array([-1, 0, 0])
        self.lft_outer.jnts[2]['loc_rotmat'] = rm.rotmat_from_euler(-49.73/180*math.pi, 0, 0)
        self.lft_outer.jnts[3]['loc_pos'] = np.array([0, 0.02752444, -0.03017323])
        self.lft_outer.jnts[3]['loc_rotmat'] = rm.rotmat_from_euler(0 ,0, 0)
        self.lft_outer.jnts[3]['loc_motionax'] = np.array([-1, 0, 0])
        # - lft_inner
        self.lft_inner = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(1), name='lft_inner')
        self.lft_inner.jnts[1]['loc_pos'] = np.array([0, -.016, .10086])
        self.lft_inner.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(49.73/180*math.pi+math.pi/2, 0, 0)
        self.lft_inner.jnts[1]['loc_motionax'] = np.array([-1, 0, 0])
        # - lft_inner_connectingrod
        self.lft_inner_connectingrod = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(3), name='lft_inner_connectingrod')
        self.lft_inner_connectingrod.jnts[1]['loc_pos'] = np.array([0, -.016, .10086])
        self.lft_inner_connectingrod.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(math.pi+38.21/180*math.pi,0,0)
        self.lft_inner_connectingrod.jnts[1]['loc_motionax'] = np.array([-1,0,0])
        self.lft_inner_connectingrod.jnts[2]['loc_pos'] = np.array([0, .037, 0])
        self.lft_inner_connectingrod.jnts[2]['loc_rotmat'] = rm.rotmat_from_euler(-95.377/180*math.pi, 0, 0)
        self.lft_inner_connectingrod.jnts[2]['loc_motionax'] = np.array([-1, 0, 0])
        self.lft_inner_connectingrod.jnts[3]['loc_pos'] = np.array([0, .0, 0])
        self.lft_inner_connectingrod.jnts[3]['loc_rotmat'] = rm.rotmat_from_euler(-174.62/180*math.pi, 0, 0)
        self.lft_inner_connectingrod.jnts[3]['loc_motionax'] = np.array([-1, 0, 0])
        # - rgt_outer
        self.rgt_outer = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(3), name='rgt_outer')
        self.rgt_outer.jnts[1]['loc_pos'] = np.array([0, .03675, .09334])
        self.rgt_outer.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(49.73/180*math.pi+math.pi/2, 0, math.pi)
        self.rgt_outer.jnts[1]['loc_motionax'] = np.array([-1, 0, 0])
        self.rgt_outer.jnts[2]['loc_pos'] = np.array([0, 0.0865, 0])  # passive
        self.rgt_outer.jnts[2]['loc_motionax'] = np.array([-1, 0, 0])
        self.rgt_outer.jnts[2]['loc_rotmat'] = rm.rotmat_from_euler(-49.73/180*math.pi, 0, 0)
        self.rgt_outer.jnts[3]['loc_pos'] = np.array([0, 0.02752444, -0.03017323])
        self.rgt_outer.jnts[3]['loc_rotmat'] = rm.rotmat_from_euler(0 ,0, 0)
        self.rgt_outer.jnts[3]['loc_motionax'] = np.array([-1, 0, 0])
        # - rgt_inner
        self.rgt_inner = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(1), name='rgt_inner')
        self.rgt_inner.jnts[1]['loc_pos'] = np.array([0, .016, .10086])
        self.rgt_inner.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(49.73/180*math.pi+math.pi/2, 0, math.pi)
        self.rgt_inner.jnts[1]['loc_motionax'] = np.array([-1, 0, 0])
        # - rgt_inner_connectingrod
        self.rgt_inner_connectingrod = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(3), name='rgt_inner_connectingrod')
        self.rgt_inner_connectingrod.jnts[1]['loc_pos'] = np.array([0, .016, .10086])
        self.rgt_inner_connectingrod.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(math.pi+38.21/180*math.pi,0,math.pi)
        self.rgt_inner_connectingrod.jnts[1]['loc_motionax'] = np.array([-1,0,0])
        self.rgt_inner_connectingrod.jnts[2]['loc_pos'] = np.array([0, .037, 0])
        self.rgt_inner_connectingrod.jnts[2]['loc_rotmat'] = rm.rotmat_from_euler(-95.377/180*math.pi, 0, 0)
        self.rgt_inner_connectingrod.jnts[2]['loc_motionax'] = np.array([-1, 0, 0])
        self.rgt_inner_connectingrod.jnts[3]['loc_pos'] = np.array([0, .0, 0])
        self.rgt_inner_connectingrod.jnts[3]['loc_rotmat'] = rm.rotmat_from_euler(-174.62/180*math.pi, 0, 0)
        self.rgt_inner_connectingrod.jnts[3]['loc_motionax'] = np.array([-1, 0, 0])
        # links
        # - lft_outer
        self.lft_outer.lnks[0]['name'] = "base"
        self.lft_outer.lnks[0]['loc_pos'] = np.zeros(3)
        self.lft_outer.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "A.stl")
        self.lft_outer.lnks[0]['rgba'] = [.2, .2, .2, 1]
        self.lft_outer.lnks[1]['name'] = "left_outer_knuckle"
        self.lft_outer.lnks[1]['loc_pos'] = np.zeros(3)
        self.lft_outer.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "H.stl")
        self.lft_outer.lnks[1]['rgba'] = [0.792156862745098, 0.819607843137255, 0.933333333333333, 1]
        self.lft_outer.lnks[2]['name'] = "left_outer_finger"
        self.lft_outer.lnks[2]['loc_pos'] = np.zeros(3)
        self.lft_outer.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "C1.stl")
        self.lft_outer.lnks[2]['rgba'] = [.2, .2, .2, 1]
        self.lft_outer.lnks[3]['name'] = "left_inner_finger"
        self.lft_outer.lnks[3]['loc_pos'] = np.zeros(3)
        self.lft_outer.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "I.stl")
        self.lft_outer.lnks[3]['rgba'] = [0,0,1, 1]
        # - lft_inner
        self.lft_inner.lnks[1]['name'] = "left_inner_knuckle"
        self.lft_inner.lnks[1]['loc_pos'] = np.zeros(3)
        self.lft_inner.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "D.stl")
        self.lft_inner.lnks[1]['rgba'] = [.2, .2, .2, 1]
        # - lft_inner_connectingrod
        self.lft_inner_connectingrod.lnks[1]['name'] = "lft_inner_connectingrod1"
        self.lft_inner_connectingrod.lnks[1]['loc_pos'] = np.zeros(3)
        self.lft_inner_connectingrod.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "F.stl")
        self.lft_inner_connectingrod.lnks[1]['rgba'] = [0.792156862745098, 0.819607843137255, 0.933333333333333, 1]
        self.lft_inner_connectingrod.lnks[2]['name'] = "lft_inner_connectingrod2"
        self.lft_inner_connectingrod.lnks[2]['loc_pos'] = np.zeros(3)
        self.lft_inner_connectingrod.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "E.stl")
        self.lft_inner_connectingrod.lnks[2]['rgba'] = [.2, .2, .2, 1]
        self.lft_inner_connectingrod.lnks[3]['name'] = "lft_inner_connectingrod3"
        self.lft_inner_connectingrod.lnks[3]['loc_pos'] = np.zeros(3)
        self.lft_inner_connectingrod.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "G.stl")
        self.lft_inner_connectingrod.lnks[3]['rgba'] = [.2, .2, .2, 1]
        # - rgt_outer
        self.rgt_outer.lnks[1]['name'] = "right_outer_knuckle"
        self.rgt_outer.lnks[1]['loc_pos'] = np.zeros(3)
        self.rgt_outer.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "H.stl")
        self.rgt_outer.lnks[1]['rgba'] = [0.792156862745098, 0.819607843137255, 0.933333333333333, 1]
        self.rgt_outer.lnks[2]['name'] = "right_outer_finger"
        self.rgt_outer.lnks[2]['loc_pos'] = np.zeros(3)
        self.rgt_outer.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "C1.stl")
        self.rgt_outer.lnks[2]['rgba'] = [.2, .2, .2, 1]
        self.rgt_outer.lnks[3]['name'] = "right_inner_finger"
        self.rgt_outer.lnks[3]['loc_pos'] = np.zeros(3)
        self.rgt_outer.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "I.stl")
        self.rgt_outer.lnks[3]['rgba'] = [0,0,1, 1]
        # - rgt_inner
        self.rgt_inner.lnks[1]['name'] = "right_inner_knuckle"
        self.rgt_inner.lnks[1]['loc_pos'] = np.zeros(3)
        self.rgt_inner.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "D.stl")
        self.rgt_inner.lnks[1]['rgba'] = [.2, .2, .2, 1]
        # - rgt_inner_connectingrod
        self.rgt_inner_connectingrod.lnks[1]['name'] = "rgt_inner_connectingrod1"
        self.rgt_inner_connectingrod.lnks[1]['loc_pos'] = np.zeros(3)
        self.rgt_inner_connectingrod.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "F.stl")
        self.rgt_inner_connectingrod.lnks[1]['rgba'] = [0.792156862745098, 0.819607843137255, 0.933333333333333, 1]
        self.rgt_inner_connectingrod.lnks[2]['name'] = "rgt_inner_connectingrod2"
        self.rgt_inner_connectingrod.lnks[2]['loc_pos'] = np.zeros(3)
        self.rgt_inner_connectingrod.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "E.stl")
        self.rgt_inner_connectingrod.lnks[2]['rgba'] = [.2, .2, .2, 1]
        self.rgt_inner_connectingrod.lnks[3]['name'] = "rgt_inner_connectingrod3"
        self.rgt_inner_connectingrod.lnks[3]['loc_pos'] = np.zeros(3)
        self.rgt_inner_connectingrod.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "G.stl")
        self.rgt_inner_connectingrod.lnks[3]['rgba'] = [.2, .2, .2, 1]
        # reinitialize
        self.lft_outer.reinitialize()
        self.lft_inner.reinitialize()
        self.lft_inner_connectingrod.reinitialize()
        self.rgt_outer.reinitialize()
        self.rgt_inner.reinitialize()
        self.rgt_inner_connectingrod.reinitialize()

        # jaw width
        self.jawwidth_rng = [0, .145]
        # jaw center
        self.jaw_center_pos = np.array([0, 0, .18169])  # position for initial state (fully open)
        # relative jaw center pos
        self.jaw_center_pos_rel = self.jaw_center_pos - self.lft_outer.jnts[3]['loc_pos']
        # collision detection
        self.all_cdelements = []
        self.enable_cc(toggle_cdprimit=enable_cc)

    def enable_cc(self, toggle_cdprimit):
        if toggle_cdprimit:
            super().enable_cc()
            # cdprimit
            self.cc.add_cdlnks(self.lft_outer, [0, 1, 2, 3])
            self.cc.add_cdlnks(self.lft_inner, [1])
            self.cc.add_cdlnks(self.rgt_outer, [1, 2, 3])
            self.cc.add_cdlnks(self.rgt_inner, [1])
            self.cc.add_cdlnks(self.lft_inner_connectingrod, [1,2,3])
            self.cc.add_cdlnks(self.rgt_inner_connectingrod, [1,2,3])
            activelist = [self.lft_outer.lnks[0],
                          self.lft_outer.lnks[1],
                          self.lft_outer.lnks[2],
                          self.lft_outer.lnks[3],
                          self.lft_inner.lnks[1],
                          self.rgt_outer.lnks[1],
                          self.rgt_outer.lnks[2],
                          self.rgt_outer.lnks[3],
                          self.rgt_inner.lnks[1],
                          self.rgt_inner_connectingrod.lnks[1],
                          self.rgt_inner_connectingrod.lnks[2],
                          self.rgt_inner_connectingrod.lnks[3],
                          self.lft_inner_connectingrod.lnks[1],
                          self.lft_inner_connectingrod.lnks[2],
                          self.lft_inner_connectingrod.lnks[3]]
            self.cc.set_active_cdlnks(activelist)
            self.all_cdelements = self.cc.all_cdelements
        # cdmesh
        for cdelement in self.all_cdelements:
            cdmesh = cdelement['collision_model'].copy()
            self.cdmesh_collection.add_cm(cdmesh)

    def fix_to(self, pos, rotmat, angle=None):
        self.pos = pos
        self.rotmat = rotmat
        if angle is not None:
            self.lft_outer.jnts[2]['motion_val'] = self.lft_outer.jnts[1]['motion_val']
            self.lft_inner.jnts[1]['motion_val'] = -self.lft_outer.jnts[1]['motion_val']
            self.rgt_outer.jnts[1]['motion_val'] = -self.lft_outer.jnts[1]['motion_val']
            self.rgt_outer.jnts[2]['motion_val'] = self.lft_outer.jnts[1]['motion_val']
            self.rgt_inner.jnts[1]['motion_val'] = -self.lft_outer.jnts[1]['motion_val']
        self.coupling.fix_to(self.pos, self.rotmat)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        self.lft_outer.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.lft_inner.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.rgt_inner.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.rgt_outer.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.lft_inner_connectingrod.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.rgt_inner_connectingrod.fix_to(cpl_end_pos, cpl_end_rotmat)

    def calculat_inner_connectingrod_jnt2(self,rad):
        self.length_virtually = math.sqrt(12 * 12 + 86.5 * 86.5 - 2 * 12 * 86.5 * math.cos(((119.73 - rad / math.pi * 180)) / 180 * math.pi))
        self.act_jnt2 = math.acos((88.9*88.9+37*37-self.length_virtually*self.length_virtually)/2/88.9/37)
        self.turn_jnt2 = 84.623/180*math.pi - self.act_jnt2
        return self.turn_jnt2

    def calculat_inner_connectingrod_jnt1(self,rad):
        self.length_virtually = math.sqrt(12 * 12 + 86.5 * 86.5 - 2 * 12 * 86.5 * math.cos(((119.73 - rad / math.pi * 180)) / 180 * math.pi))
        self.act_jnt2 = math.acos((88.9*88.9+37*37-self.length_virtually*self.length_virtually)/2/88.9/37)
        self.virtually_jnt1 = math.asin(88.9 * math.sin(self.act_jnt2)/self.length_virtually)
        self.transition_jnt1 = math.asin(12 * math.sin(((119.73 - rad / math.pi * 180)) / 180 * math.pi)/self.length_virtually)
        self.act_jnt1 = self.virtually_jnt1 - (math.pi-self.transition_jnt1-((119.73 - rad / math.pi * 180)) / 180  * math.pi)
        self.turn_jnt1 = 18.21/180*math.pi - self.act_jnt1
        return self.turn_jnt1

    def calculat_inner_connectingrod_jnt3(self,rad):
        self.length_virtually = math.sqrt(12 * 12 + 86.5 * 86.5 - 2 * 12 * 86.5 * math.cos(((119.73 - rad / math.pi * 180)) / 180 * math.pi))
        self.act_jnt2 = math.acos((88.9*88.9+37*37-self.length_virtually*self.length_virtually)/2/88.9/37)
        self.virtually_jnt1 = math.asin(88.9 * math.sin(self.act_jnt2)/self.length_virtually)
        self.transition_jnt1 = math.asin(12 * math.sin(((119.73 - rad / math.pi * 180)) / 180 * math.pi)/self.length_virtually)
        self.act_jnt1 = self.virtually_jnt1 - (math.pi-self.transition_jnt1-((119.73 - rad / math.pi * 180)) / 180  * math.pi)
        self.cal_jnt1 = -20/180*math.pi - self.act_jnt1
        self.side = math.cos(self.cal_jnt1)*0.037 - 0.005745
        self.transition_jnt3 = math.cos(self.side/0.03771)
        self.act_jnt3 = self.transition_jnt3 - self.cal_jnt1 + self.act_jnt2
        self.turn_jnt3 = -174.62/180*math.pi + self.act_jnt3
        return self.turn_jnt3

    def fk(self, motion_val):
        """
        lft_outer is the only active joint, all others mimic this one
        :param: angle, radian
        """
        if self.lft_outer.jnts[1]['motion_rng'][0] <= motion_val <= self.lft_outer.jnts[1]['motion_rng'][1]:
            self.lft_outer.jnts[1]['motion_val'] = motion_val
            self.lft_outer.jnts[2]['motion_val'] = -self.lft_outer.jnts[1]['motion_val']
            self.lft_inner.jnts[1]['motion_val'] = self.lft_outer.jnts[1]['motion_val']
            self.rgt_outer.jnts[1]['motion_val'] = self.lft_outer.jnts[1]['motion_val']
            self.rgt_outer.jnts[2]['motion_val'] = -self.lft_outer.jnts[1]['motion_val']
            self.rgt_inner.jnts[1]['motion_val'] = self.lft_outer.jnts[1]['motion_val']
            self.lft_inner_connectingrod.jnts[1]['motion_val'] = self.calculat_inner_connectingrod_jnt1(self.lft_inner.jnts[1]['motion_val'])
            self.lft_inner_connectingrod.jnts[2]['motion_val'] = self.calculat_inner_connectingrod_jnt2(self.lft_inner.jnts[1]['motion_val'])
            self.lft_inner_connectingrod.jnts[3]['motion_val'] = self.calculat_inner_connectingrod_jnt3(self.lft_inner.jnts[1]['motion_val'])
            self.rgt_inner_connectingrod.jnts[1]['motion_val'] = self.lft_inner_connectingrod.jnts[1]['motion_val']
            self.rgt_inner_connectingrod.jnts[2]['motion_val'] = self.lft_inner_connectingrod.jnts[2]['motion_val']
            self.rgt_inner_connectingrod.jnts[3]['motion_val'] = self.lft_inner_connectingrod.jnts[3]['motion_val']
            self.lft_outer.fk()
            self.lft_inner.fk()
            self.rgt_outer.fk()
            self.rgt_inner.fk()
            self.lft_inner_connectingrod.fk()
            self.rgt_inner_connectingrod.fk()
        else:
            raise ValueError("The angle parameter is out of range!")

    def _from_distance_to_radians(self, distance):
        """
        private helper function to convert a command in meters to radians (joint value)
        """
        # return np.clip(
        #   self.lft_outer.jnts[1]['motion_rng'][1] - ((self.lft_outer.jnts[1]['motion_rng'][1]/self.jawwidth_rng[1]) * distance),
        #   self.lft_outer.jnts[1]['motion_rng'][0], self.lft_outer.jnts[1]['motion_rng'][1]) # kiyokawa, commented out by weiwei
        return np.clip(self.lft_outer.jnts[1]['motion_rng'][1] - math.asin(
            (math.sin(self.lft_outer.jnts[1]['motion_rng'][1]) / self.jawwidth_rng[1]) * distance),
                       self.lft_outer.jnts[1]['motion_rng'][0], self.lft_outer.jnts[1]['motion_rng'][1])

    def jaw_to(self, jaw_width):
        # print(jaw_width)
        if jaw_width > self.jawwidth_rng[1]:
            raise ValueError(f"Jawwidth must be {self.jawwidth_rng[0]}mm~{self.jawwidth_rng[1]}mm!")
        motion_val = self._from_distance_to_radians(jaw_width)
        self.fk(motion_val)
        # TODO dynamically change jaw center
        # print(self.jaw_center_pos_rel)
        self.jaw_center_pos=np.array([0,0,self.lft_outer.jnts[3]['loc_pos'][2]+self.jaw_center_pos_rel[2]])

    def gen_stickmodel(self,
                       tcp_jntid=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='ag145_stickmodel'):
        sm_collection = mc.ModelCollection(name=name)
        self.coupling.gen_stickmodel(toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs).attach_to(sm_collection)
        self.lft_outer.gen_stickmodel(toggle_tcpcs=False,
                                      toggle_jntscs=toggle_jntscs,
                                      toggle_connjnt=toggle_connjnt).attach_to(sm_collection)
        self.lft_inner.gen_stickmodel(toggle_tcpcs=False,
                                      toggle_jntscs=toggle_jntscs,
                                      toggle_connjnt=toggle_connjnt).attach_to(sm_collection)
        self.rgt_outer.gen_stickmodel(toggle_tcpcs=False,
                                      toggle_jntscs=toggle_jntscs,
                                      toggle_connjnt=toggle_connjnt).attach_to(sm_collection)
        self.rgt_inner.gen_stickmodel(toggle_tcpcs=False,
                                      toggle_jntscs=toggle_jntscs,
                                      toggle_connjnt=toggle_connjnt).attach_to(sm_collection)
        self.lft_inner_connectingrod.gen_stickmodel(toggle_tcpcs=False,
                                      toggle_jntscs=toggle_jntscs,
                                      toggle_connjnt=toggle_connjnt).attach_to(sm_collection)
        self.rgt_inner_connectingrod.gen_stickmodel(toggle_tcpcs=False,
                                      toggle_jntscs=toggle_jntscs,
                                      toggle_connjnt=toggle_connjnt).attach_to(sm_collection)
        if toggle_tcpcs:
            jaw_center_gl_pos = self.rotmat.dot(grpr.jaw_center_pos) + self.pos
            jaw_center_gl_rotmat = self.rotmat.dot(grpr.jaw_center_rotmat)
            gm.gen_dashstick(spos=self.pos,
                             epos=jaw_center_gl_pos,
                             thickness=.0062,
                             rgba=[.5, 0, 1, 1],
                             type="round").attach_to(sm_collection)
            gm.gen_mycframe(pos=jaw_center_gl_pos, rotmat=jaw_center_gl_rotmat).attach_to(sm_collection)
        return sm_collection

    def gen_meshmodel(self,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='ag145_meshmodel'):
        mm_collection = mc.ModelCollection(name=name)
        self.coupling.gen_meshmodel(toggle_tcpcs=False,
                                    toggle_jntscs=toggle_jntscs,
                                    rgba=rgba).attach_to(mm_collection)
        self.lft_outer.gen_meshmodel(toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs,
                                     rgba=rgba).attach_to(mm_collection)
        self.lft_inner.gen_meshmodel(toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs,
                                     rgba=rgba).attach_to(mm_collection)
        self.rgt_outer.gen_meshmodel(toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs,
                                     rgba=rgba).attach_to(mm_collection)
        self.rgt_inner.gen_meshmodel(toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs,
                                     rgba=rgba).attach_to(mm_collection)
        self.lft_inner_connectingrod.gen_meshmodel(toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs,
                                     rgba=rgba).attach_to(mm_collection)
        self.rgt_inner_connectingrod.gen_meshmodel(toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs,
                                     rgba=rgba).attach_to(mm_collection)
        if toggle_tcpcs:
            jaw_center_gl_pos = self.rotmat.dot(self.jaw_center_pos) + self.pos
            jaw_center_gl_rotmat = self.rotmat.dot(self.jaw_center_rotmat)
            gm.gen_dashstick(spos=self.pos,
                             epos=jaw_center_gl_pos,
                             thickness=.0062,
                             rgba=[.5, 0, 1, 1],
                             type="round").attach_to(mm_collection)
            gm.gen_mycframe(pos=jaw_center_gl_pos, rotmat=jaw_center_gl_rotmat).attach_to(mm_collection)
        return mm_collection

    def get_jawwidth(self):
        # return (self.lft_outer.jnts[2]['gl_posq'][2]+0.17959538)*2
        return (self.lft_outer.jnts[2]['gl_posq'][1]+0.03024929)*2

if __name__ == '__main__':
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    grpr = Ag145(enable_cc=True)
    jawwidth = 0.145
    grpr.jaw_to(jawwidth)
    print(grpr.jaw_center_pos)
    grpr.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
    print(grpr.get_jawwidth())
    gm.gen_sphere(grpr.get_jawwidth()).attach_to(base)
    # gm.gen_stick(np.array([0, 0, 0]), np.array([0, 0, 100])).attach_to(base)

    base.run()

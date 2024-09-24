import itertools
import pickle
import os
# import environment.bulletcdhelper as bch
# import envloader as el
import basis.robot_math as rm
import modeling.collision_model as cm
import numpy as np
import loadhelper as el

class HandoverMaster(object):
    """

    author: hao chen, ruishuang liu, refactored by weiwei
    date: 20191122
    """

    def __init__(self, obj, rbthi, retractdistance=100.0):
        """

        :param obj: obj name (str) or objcm, objcm is for debug purpose
        :param rhx: see helper.py
        :param retractdistance: retraction distance

        author: hao, refactored by weiwei
        date: 20191206, 20200104osaka
        """

        if isinstance(obj, str):
            self.objname = obj
        elif isinstance(obj, cm.CollisionModel):
            self.objname = obj.name
            self.objcm = obj
        self.rbthi = rbthi
        # self.rbt = rbthi.rbt
        self.retractdistance = retractdistance
        # self.bcdchecker = bch.MCMchecker(toggledebug=False)
        self.rgthndfa = rbthi.rgt_hnd
        self.lfthndfa = rbthi.lft_hnd

        # with open(os.path.join(el.root, "grasp_" + self.rgthndfa.name, "predefinedgrasps.pickle"), "rb") as file:
        with open(os.path.join(el.root, "test_long_grasp.pickle"), "rb") as file:
            graspdata = pickle.load(file)
            # if self.objname in graspdata.keys():
            #     self.identityglist_rgt = graspdata[self.objname]
            # else:
            #     self.identityglist_rgt = []
            self.identityglist_rgt = graspdata
        # with open(os.path.join(el.root, "grasp_" + self.lfthndfa.name, "predefinedgrasps.pickle"), "rb") as file:
        with open(os.path.join(el.root, "test_long_grasp.pickle"), "rb") as file:
            graspdata = pickle.load(file)
            # if self.objname in graspdata.keys():
            #     self.identityglist_lft = graspdata[self.objname]
            # else:
            #     self.identityglist_lft = []
            self.identityglist_lft = graspdata
        self.grasp = [self.identityglist_rgt,self.identityglist_lft]
        self.hndfa = [self.rgthndfa, self.lfthndfa]

        # paramters
        self.fpsnpmat4 = []
        self.identitygplist = [] # grasp pair list at the identity pose
        self.fpsnestedglist_rgt = {} # fpsnestedglist_rgt[fpid] = [g0, g1, ...], fpsnestedglist means glist at each floating pose
        self.fpsnestedglist_lft = {} # fpsnestedglist_lft[fpid] = [g0, g1, ...]
        self.ikfid_fpsnestedglist_rgt = {} # fid - feasible id
        self.ikfid_fpsnestedglist_lft = {}
        self.ikjnts_fpsnestedglist_rgt = {}
        self.ikjnts_fpsnestedglist_lft = {}

    def genhvgpsgl(self, posvec, rotmat = None , debug = False):
        """
        generate the handover grasps using the given position and orientation
        sgl means a single position
        rotmat could either be a single one or multiple (0,90,180,270, default)

        :param posvec
        :param rotmat
        :return: data is saved as a file

        author: hao chen, refactored by weiwei  7020397
        date: 20191122
        """

        self.identitygplist = []
        if rotmat is None:
            # self.fpsnpmat4 = rm.gen_icohomomats_flat(posvec=posvec, angles=[0,45,90,135,180,225,270])
            self.fpsnpmat4 = rm.gen_icohomomats(position=posvec, rotagls=[0])
        else:
            self.fpsnpmat4 = [rm.homomat_from_posrot(posvec, rotmat)]
        if debug:
            import copy
            for mat in self.fpsnpmat4:
                objtmp = copy.deepcopy(self.objcm)
                objtmp.set_homomat(mat)
                objtmp.set_rgba([0, 191/255,1,0.3])
                objtmp.attach_to(base)

            base.run()

        self.__genidentitygplist()
        self.__genfpsnestedglist()
        self.__checkik()

        if not os.path.exists(os.path.join(el.root, "data_handover")):
            os.mkdir(os.path.join(el.root, "data_handover"))
        with open(os.path.join(el.root, "data_handover", self.objname + "_hndovrinfo.pickle"), "wb") as file:
            pickle.dump([self.fpsnpmat4, self.identitygplist, self.fpsnestedglist_rgt, self.fpsnestedglist_lft,
                         self.ikfid_fpsnestedglist_rgt, self.ikfid_fpsnestedglist_lft,
                         self.ikjnts_fpsnestedglist_rgt, self.ikjnts_fpsnestedglist_lft], file)

    def genhvgplist(self, hvgplist):
        """
        generate the handover grasps using the given list of homomat

        :param hvgplist, [homomat0, homomat1, ...]
        :return: data is saved as a file

        author: hao chen, refactored by weiwei
        date: 20191122
        """

        self.identitygplist = []
        self.fpsnpmat4 = hvgplist
        self.__genidentitygplist()
        self.__genfpsnestedglist()
        self.__checkik()

        if not os.path.exists(os.path.join(el.root, "data_handover")):
            os.mkdir(os.path.join(el.root, "data_handover"))
        with open(os.path.join(el.root, "data_handover", self.objname + "_hndovrinfo.pickle"), "wb") as file:
            pickle.dump([self.fpsnpmat4, self.identitygplist, self.fpsnestedglist_rgt, self.fpsnestedglist_lft,
                         self.ikfid_fpsnestedglist_rgt, self.ikfid_fpsnestedglist_lft,
                         self.ikjnts_fpsnestedglist_rgt, self.ikjnts_fpsnestedglist_lft], file)

    def gethandover(self):
        """
        io interface to load the previously planned data

        :return:

        author: hao, refactored by weiwei
        date: 20191206, 20191212
        """

        with open(os.path.join(el.root, "data_handover", self.objname + "_hndovrinfo.pickle"), "rb") as file:
            self.fpsnpmat4, self.identitygplist, self.fpsnestedglist_rgt, self.fpsnestedglist_lft, \
            self.ikfid_fpsnestedglist_rgt, self.ikfid_fpsnestedglist_lft, \
            self.ikjnts_fpsnestedglist_rgt, self.ikjnts_fpsnestedglist_lft = pickle.load(file)

        return self.identityglist_rgt, self.identityglist_lft, self.fpsnpmat4, \
               self.identitygplist, self.fpsnestedglist_rgt, self.fpsnestedglist_lft, \
               self.ikfid_fpsnestedglist_rgt, self.ikfid_fpsnestedglist_lft, \
               self.ikjnts_fpsnestedglist_rgt, self.ikjnts_fpsnestedglist_lft

    def __genidentitygplist(self):
        """
        fill up self.identitygplist

        :return:

        author: weiwei
        date: 20191212
        """

        rgthnd = self.rgthndfa
        lfthnd = self.lfthndfa
        pairidlist = list(itertools.product(range(len(self.identityglist_rgt)), range(len(self.identityglist_lft))))
        for i in range(len(pairidlist)):
            print("generating identity gplist...", i, len(pairidlist))
            # Check whether the hands collide with each or not
            ir, il = pairidlist[i]
            rgthnd.setMat(base.pg.np4ToMat4(self.identityglist_rgt[ir][2]))
            rgthnd.setjawwidth(self.identityglist_rgt[ir][0])
            lfthnd.setMat(base.pg.np4ToMat4(self.identityglist_lft[il][2]))
            lfthnd.setjawwidth(self.identityglist_lft[il][0])
            ishndcollided = rgthnd.is_mesh_collided(lfthnd.cmlist)
            # ishndcollided = self.bcdchecker.isMeshListMeshListCollided(rgthnd.cmlist, lfthnd.cmlist)
            if not ishndcollided:
                self.identitygplist.append(pairidlist[i])

    def __genfpsnestedglist(self):
        """
        generate the grasp list for the floating poses

        :return:

        author: hao chen, revised by weiwei
        date: 20191122
        """

        self.fpsnestedglist_rgt = {}
        self.fpsnestedglist_lft = {}
        for posid, icomat4 in enumerate(self.fpsnpmat4):
            print("generating nested glist at the floating poses...", posid, len(self.fpsnpmat4))
            glist = []
            for jawwidth, fc, homomat, approach_direction in self.identityglist_rgt:
                tippos =  rm.homomat_transform_points(icomat4, fc)
                homomat = np.dot(icomat4, homomat)
                approach_direction = np.dot(icomat4[:3,:3], approach_direction)
                glist.append([jawwidth, tippos, homomat, approach_direction])
            self.fpsnestedglist_rgt[posid] = glist
            glist = []
            for jawwidth, fc, homomat, approach_direction in self.identityglist_lft:
                tippos =  rm.homomat_transform_points(icomat4, fc)
                homomat = np.dot(icomat4, homomat)
                approach_direction = np.dot(icomat4[:3, :3], approach_direction)
                glist.append([jawwidth, tippos, homomat, approach_direction])
            self.fpsnestedglist_lft[posid] = glist

    def __checkik(self):
        # Check the IK of both hand in the handover pose
        ### right hand
        self.ikfid_fpsnestedglist_rgt = {}
        self.ikjnts_fpsnestedglist_rgt = {}
        self.ikfid_fpsnestedglist_lft = {}
        self.ikjnts_fpsnestedglist_lft = {}
        for posid in self.fpsnestedglist_rgt.keys():
            armname = 'rgt'
            fpglist_thispose = self.fpsnestedglist_rgt[posid]
            for i, [_, tippos, homomat, handa] in enumerate(fpglist_thispose):
                hndrotmat4 = homomat
                fgrcenternp = tippos
                fgrcenterrotmatnp = hndrotmat4[:3, :3]
                handa = -handa
                minusworldy = np.array([0,-1,0])
                if rm.angle_between_vectors(handa,minusworldy) < np.pi*0.5:
                    msc = self.rbt.numik(fgrcenternp, fgrcenterrotmatnp, armname)
                    if msc is not None:
                        fgrcenternp_handa = fgrcenternp + handa * self.retractdistance
                        msc_handa = self.rbt.numikmsc(fgrcenternp_handa, fgrcenterrotmatnp, msc, armname)
                        if msc_handa is not None:
                            if posid not in self.ikfid_fpsnestedglist_rgt:
                                self.ikfid_fpsnestedglist_rgt[posid] = []
                            self.ikfid_fpsnestedglist_rgt[posid].append(i)
                            if posid not in self.ikjnts_fpsnestedglist_rgt:
                                self.ikjnts_fpsnestedglist_rgt[posid] = {}
                            self.ikjnts_fpsnestedglist_rgt[posid][i] = [msc, msc_handa]

        ### left hand
        for posid in self.fpsnestedglist_lft.keys():
            armname = 'lft'
            fpglist_thispose = self.fpsnestedglist_lft[posid]
            for i, [_, tippos, homomat, handa] in enumerate(fpglist_thispose):
                hndrotmat4 = homomat
                fgrcenternp = tippos
                fgrcenterrotmatnp = hndrotmat4[:3, :3]
                handa = -handa
                plusworldy = np.array([0,1,0])
                if rm.angle_between_vectors(handa,plusworldy) < np.pi*0.5:
                    msc = self.rbt.numik(fgrcenternp, fgrcenterrotmatnp, armname)
                    if msc is not None:
                        fgrcenternp_handa = fgrcenternp + handa * self.retractdistance
                        msc_handa = self.rbt.numikmsc(fgrcenternp_handa, fgrcenterrotmatnp, msc, armname)
                        if msc_handa is not None:
                            if posid not in self.ikfid_fpsnestedglist_lft:
                                self.ikfid_fpsnestedglist_lft[posid] = []
                            self.ikfid_fpsnestedglist_lft[posid].append(i)
                            if posid not in self.ikjnts_fpsnestedglist_lft:
                                self.ikjnts_fpsnestedglist_lft[posid] = {}
                            self.ikjnts_fpsnestedglist_lft[posid][i] = [msc, msc_handa]


    def showIKFandCFgrasp(self, homomat, obscmlist, graspid = -1, armname = "rgt"):
        if armname == "rgt":
            predefinedgrasps = self.grasp[0]
            hndfa = self.rgthndfa
        else:
            predefinedgrasps = self.grasp[1]
            hndfa = self.lfthndfa
        if graspid > 0:
            predefinedgrasps = [predefinedgrasps[graspid]]
        availableHnd = []
        availableID = []
        collidednum = 0
        infeasiblenum = 0
        print(f"Number of grasps candidate: {len(predefinedgrasps)}")
        for idpre, predefined_grasp in enumerate(predefinedgrasps):
            if len(predefined_grasp) == 3:
                prejawwidth, prehndfc, prehomomat = predefined_grasp
            else:
                prejawwidth, prehndfc, prehomomat, _ = predefined_grasp
            hndmat4 = np.dot(homomat, prehomomat)
            eepos = rm.homomat_transform_points(homomat, prehndfc)[:3]
            eerot = hndmat4[:3, :3]

            hndnew = hndfa.genHand()
            hndnew.sethomomat(hndmat4)
            hndnew.setjawwidth(prejawwidth)
            isHndCollided = self.bcdchecker.isMeshListMeshListCollided(hndnew.cmlist, obscmlist)
            if not isHndCollided:
                armjnts = self.rbt.numik(eepos, eerot, armname)
                if armjnts is not None:
                    hndnew = hndfa.genHand()
                    hndnew.setColor(0, 1, 0, .5)
                    hndnew.sethomomat(hndmat4)
                    hndnew.setjawwidth(prejawwidth)
                    availableHnd.append(hndnew)
                    availableID.append(idpre)
                else:
                    infeasiblenum += 1
            else:
                collidednum += 1
                # bcdchecker.showMeshList(hndnew.cmlist)
        print(f"Number of collided grasps {collidednum}")
        print(f"number of IK-infeasible grasps {infeasiblenum}")
        print("The useful id is: ", availableID)
        return availableHnd
    def checkhndenvcollision(self, homomat, obstaclecmlist, armname="rgt", debug=False):
        """

        :param homomat:
        :param obstaclecmlist:
        :return:

        author: ruishuang
        date: 20191122
        """

        if armname == "rgt":
            handtmp = self.rgthndfa.genHand()
        else:
            handtmp = self.lfthndfa.genHand()
        handtmp.sethomomat(homomat)
        handtmp.setjawwidth(handtmp.jawwidthopen)
        iscollided = self.bcdchecker.isMeshListMeshListCollided(handtmp.cmlist, obstaclecmlist)
        if debug:
            if iscollided:
                handtmp.setColor(1,0,0,.2)
                handtmp.reparentTo(base.render)
                # base.run()
            else:
                handtmp.setColor(0, 1, 0, .2)
                handtmp.reparentTo(base.render)

        return iscollided

if __name__ == "__main__":

    import robot_sim.end_effectors.grippers.robotiqhe.robotiqhe as rtqhe

    base = el.loadEnv()

    # objname = 'motor'
    objname = 'lshape'

    rgthndfa = rtqhe.RobotiqHE(enable_cc=True)
    lfthndfa = rtqhe.RobotiqHE(enable_cc=True)
    rbthi, rbtmg = el.loadRbt()
    rbtcm = rbtmg
    rbtmg.attach_to(base)
    objcm = cm.CollisionModel(f"./objects/{objname}.stl")
    hmstr = HandoverMaster(obj=objcm, rbthi=rbthi, retractdistance=0.060)
    hmstr.genhvgpsgl(np.array([.700, 0, 1.300]), debug=False)




    # identityglist_rgt: grasp of the right hand
    # identityglist_lft: grasp of the left hand
    # fpsnpmat4: 4x4 matrix of the object
    # identitygplist:

    # identityglist_rgt, identityglist_lft, fpsnpmat4, identitygplist, fpsnestedglist_rgt, fpsnestedglist_lft, \
    # ikfid_fpsnestedglist_rgt,ikfid_fpsnestedglist_lft, \
    # ikjnts_fpsnestedglist_rgt, ikjnts_fpsnestedglist_lft = hmstr.gethandover()
    # print(ikfid_fpsnestedglist_lft.keys())
    # print(ikfid_fpsnestedglist_rgt.keys())
    # # #
    # poseid = 0
    # for gp in identitygplist:
    #     ir = gp[0]
    #     il = gp[1]
    #     # check if it possible for both left arm and right hand move to the handover position
    #     if poseid not in ikfid_fpsnestedglist_lft.keys() or poseid not in ikfid_fpsnestedglist_rgt:
    #         poseid += 1
    #         print("Poseid " + str(poseid) + " do not have feasible ik solution, continue...")
    #         continue
    #     if ir in ikfid_fpsnestedglist_rgt[poseid] and il in ikfid_fpsnestedglist_lft[poseid]:
    #         jawwidth_rgt, fc_rgt, rotmat_rgt = fpsnestedglist_rgt[poseid][ir]
    #         jnts_rgt, jnts_msc_rgt = ikjnts_fpsnestedglist_rgt[poseid][ir]
    #         jawwidth_lft, fc_lft, rotmat_lft = fpsnestedglist_rgt[poseid][il]
    #         jnts_lft, jnts_msc_lft = ikjnts_fpsnestedglist_lft[poseid][il]
    #
    #         rbt.movearmfk(jnts_rgt, armname="rgt")
    #         rbt.movearmfk(jnts_lft, armname="lft")
    #         rbt.opengripper(jawwidth=jawwidth_rgt, armname="rgt")
    #         rbt.opengripper(jawwidth=jawwidth_lft, armname="lft")
    #         # rbtmesh.genmnp(rbt).reparentTo(base.render)
    #

import copy
import math
import visualization.panda.world as wd
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.yumi_gripper.yumi_gripper as yg
import robot_sim.end_effectors.gripper.robotiqhe.robotiqhe as hnde
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
import robot_sim.end_effectors.gripper.robotiqhe.robotiqhe as rtqhe
# import slope
# import Sptpolygoninfo as sinfo
import basis.trimesh as trimeshWan
import trimesh as trimesh
from panda3d.core import NodePath
import open3d as o3d
import vision.depth_camera.pcd_data_adapter as vdda

class TrimeshHu(object):
    def __init__(self, meshpath = None, name = None, mesh = None, scale = 1.0):

        if not mesh:
            self.name = name
            self.meshpath = meshpath +'/'+ name
            self.mesh = trimesh.load(self.meshpath)
        else:
            self.mesh = mesh
        self.mesh.fix_normals()
        self.__infoUpdate(self.mesh)

        self.originalmesh = trimesh.Trimesh(vertices=self.vertices*scale, faces=self.faces*scale, face_normals=self.face_normals,
                                       vertex_normals=self.vertex_normals)
        self.mesh = copy.copy(self.originalmesh)
        self.scalerate = 50
        self.blocksize = self.set_blocksize(self.scalerate)
        # self.mesh.scale(scale)
    # def set_scale

    def get_bb(self, show = True, option = "aabb"):
        # to_origin, extents = trimesh.bounds.oriented_bounds(self.originalmesh, angle_digits=1, ordered=True, normal=None)
        # extents = self.mesh.extents
        extents,to_origin = trimesh.bounds.to_extents(self.mesh.bounds)
        # to_origin = np.eye(4)
        # to_origin, extents = trimesh.bounds.to_extents()
        if show:
            if option == "AABB":
                gm.gen_frame_box(extents,to_origin).attach_to(base)
            if option =="o":
                gm.gen_frame_box(extents, np.linalg.inv(to_origin)).attach_to(base)
            if option =="max":
                extents = [np.max(extents)*self.blocksize]*3
                gm.gen_frame_box(extents,to_origin).attach_to(base)
        self.boxhomo=to_origin
        return to_origin, extents

    def set_blocksize(self, rate = 50):
        _, extents = self.get_bb(show = True)
        blocksize = np.max(extents)/rate
        return blocksize

    @property
    def get_blocksize(self):
        return self.blocksize

    def voxelization(self, voxel, hollow):
        self.voxel = voxel
        if hollow == True:
            self.voxelizedmodel = self.mesh.voxelized(voxel).hollow()
            self.tfmatrix = self.voxelizedmodel.transform
            self.matrix = self.voxelizedmodel.matrix
            print(self.matrix.shape)
            self.points = self.voxelizedmodel.points
            self.mesh = self.voxelizedmodel.as_boxes()
            # self.mesh = voxelizedmodel.marching_cubes
        else:
            self.voxelizedmodel = self.mesh.voxelized(voxel).fill(method='base')
            self.tfmatrix = self.voxelizedmodel.transform
            self.matrix = self.voxelizedmodel.matrix
            self.points = self.voxelizedmodel.points
            self.mesh = self.voxelizedmodel.as_boxes()
            # self.mesh = voxelizedmodel.marching_cubes
        self.__infoUpdate(self.mesh)

        '''
        slide matrix
        '''
        # output_mat = np.zeros([self.scalerate+1,self.scalerate+1,self.scalerate+1,1])
        # mat_int = np.expand_dims(self.matrix.astype(np.int), axis = 3)
        # offset_0 = np.zeros([output_mat.shape[0]-mat_int.shape[0],mat_int.shape[1], mat_int.shape[2], 1])
        # offset_1 = np.zeros([output_mat.shape[0], output_mat.shape[1]-mat_int.shape[1], mat_int.shape[2], 1])
        # offset_2 = np.zeros([output_mat.shape[0], output_mat.shape[1], output_mat.shape[2]-mat_int.shape[2], 1])
        # mat_int = np.concatenate([mat_int, offset_0], axis = 0)
        # mat_int = np.concatenate([mat_int, offset_1], axis = 1)
        # mat_int = np.concatenate([mat_int, offset_2], axis = 2)
        # mat_int[0][0][0][0] = 1000
        # multipler = np.full((self.scalerate+1,self.scalerate+1,self.scalerate+1,1), self.blocksize)
        # output_mat = np.round(mat_int * multipler, decimals=7)
        # # print(mat_int.shape)


    def get_node_matrix(self):
        matrix = [[[[self.matrix[i][j][k]*i*self.voxel+self.tfmatrix[0][3], self.matrix[i][j][k]*j*self.voxel+self.tfmatrix[1][3], self.matrix[i][j][k]*k*self.voxel+self.tfmatrix[2][3]] for k in range(len(self.matrix[i][j]))] for j in range(len(self.matrix[i]))] for i in range(len(self.matrix))]
        # print(np.asarray(matrix, dtype=float))
        self.node_matrix = np.asarray(matrix)
        # print(self.node_matrix)
        return self.node_matrix

    def get_transform(self):
        # print(self.tfmatrix)
        return self.tfmatrix

    def show_balls(self):
        # for i_index, i in enumerate(self.node_matrix):
        #     for j_index, j in enumerate(i):
        #         for k_index, k in enumerate(j):
        #             if self.matrix[i_index][j_index][k_index]:
        #                 gm.gen_sphere(k*0.001, 0.001, [1,0,0,0.5]).attach_to(base)

        for point in self.points:
            gm.gen_sphere(point, .001, [1, 0, 0, 0.5]).attach_to(base)

    def triWantotri(self, mesh):
        faces = np.asarray(mesh.faces)
        vertices = np.asarray(mesh.vertices)
        face_normals = np.asarray(mesh.face_normals)
        vertex_normals = np.asarray(mesh.vertex_normals)
        new_mesh = trimesh.Trimesh(faces=faces, vertices=vertices, face_normals=face_normals, vertex_normals=vertex_normals)
        return new_mesh

    def show_hited_balls(self, observe_origin,target,shape = "box", generateCKlist=False):
        # for i_index, i in enumerate(self.node_matrix):
        #     for j_index, j in enumerate(i):
        #         for k_index, k in enumerate(j):
        #             if self.matrix[i_index][j_index][k_index]:
        #                 gm.gen_sphere(k*0.001, 0.001, [1,0,0,0.5]).attach_to(base)
        # self.hited_list = []
        hited_list =  self.hitray(observe_origin)

        '''
        remove outlier
        '''
        def display_inlier_outlier(cloud, ind):
            inlier_cloud = cloud.select_by_index(ind)
            outlier_cloud = cloud.select_by_index(ind, invert=True)

            print("Showing outliers (red) and inliers (gray): ")
            outlier_cloud.paint_uniform_color([1, 0, 0])
            inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
            o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                              zoom=0.3412,
                                              front=[0.4257, -0.2125, -0.8795],
                                              lookat=[0, 0, 0],
                                              up=[-0.0694, -0.9768, 0.2024])

        pcd = vdda.nparray_to_o3dpcd(np.asarray(hited_list))
        cl, ind = pcd.remove_radius_outlier(nb_points=15, radius=self.blocksize*3)
        display_inlier_outlier(pcd, ind)
        hited_list = vdda.o3dpcd_to_parray(cl)


        partialmesh = self.triWantotri(gm.gen_box(extent=[self.voxel]*3, homomat = rm.homomat_from_posrot(hited_list[0])).objtrm)
        partialmesh.export("partial_bunny.stl")
        partialmesh_cklist = []
        center = np.average(hited_list, axis = 0)

        for point in hited_list:
            if shape == "box":
                element = cm.gen_box(extent=[self.voxel*1.2]*3, homomat = rm.homomat_from_posrot(point), rgba = [1, 0, 0, 0.1])
                element.attach_to(target)
                if generateCKlist:
                    # element_mesh = self.triWantotri(element.objtrm)
                    # partialmesh = partialmesh.union([element_mesh], engine="blender")
                    # print("union")
                    # partialmesh_cklist.append(self.triWantotri(element.objtrm))
                    partialmesh_cklist.append(element)
            else:
                gm.gen_sphere(point, .001, [1, 0, 0, 0.5]).attach_to(target)

        if generateCKlist:
            return partialmesh_cklist


    def cpt_briefgrasp(self, observe_origin,target,gripper, grasp_info_list, generateCKlist=True):
        '''
        cpt = compute
        :param observe_origin:
        :param target:
        :return:
        '''

        partialmeshCKlist = self.show_hited_balls(observe_origin,target,shape = "box", generateCKlist=generateCKlist)
        # for element in partialmeshCKlist:
        #     element.attach_to(target)
        briefgrasplist = []
        voxel_grct_list = []
        direction_z_list  = []
        direction_x_list  = []
        for i, grasp_info in enumerate(grasp_info_list):
            if i %5 == 0:
                jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
                gripper.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
                # if gripper.is_mesh_collided(objcm_list=partialmeshCKlist):
                #     gripper.gen_meshmodel(rgba=(0.3, 0.3, 0.3, 0.1)).attach_to(base)
                # else:
                #     gripper.gen_meshmodel(rgba=(1, 0, 0, 0.5)).attach_to(base)
                # gm.gen_sphere(pos = jaw_center_pos, radius = 0.02).attach_to(base)
                # gm.gen_frame(pos=jaw_center_pos, rotmat= hnd_rotmat, thickness=0.02).attach_to(base)
                checker = cm.gen_stick(spos=jaw_center_pos + 0.11 * jaw_center_rotmat[:, 0], epos=jaw_center_pos - 0.11 * jaw_center_rotmat[:, 0],
                             rgba=(0.1, 0.1, 0.1, 0.1), thickness=0.001)
                voxel_grct = np.around(jaw_center_pos/self.blocksize, decimals=0, out=None)
                direction_z = hnd_rotmat[:, 2]
                direction_x = hnd_rotmat[:, 0]
                voxel_grct_list.append(voxel_grct)
                direction_z_list.append(direction_z)
                direction_x_list.append(direction_x)

                collideinfo = checker.is_mcdwith(partialmeshCKlist, toggle_contacts=True)
                if not collideinfo[0]:
                    briefgrasplist.append(grasp_info)
                    '''
                    Real color
                    '''
                    # gripper.gen_meshmodel(rgba=(0, 0, 1, 0.3)).attach_to(base)
                    # gripper.gen_meshmodel().attach_to(base)


                    # gripper.gen_meshmodel(rgba=(0.3, 0.3, 0.3, 0.3)).attach_to(base)
                    gm.gen_arrow(spos=jaw_center_pos -jaw_width * jaw_center_rotmat[:, 0]/2, epos=jaw_center_pos+jaw_width * jaw_center_rotmat[:,0]/2, rgba=(0.1,0.1,0.1,0.01), thickness=0.002).attach_to(base)
                    pass
                else:
                    '''
                    Real color
                   '''
                    # gripper.gen_meshmodel(rgba=(0, 0, 1, 0.3)).attach_to(base)
                    # gripper.gen_meshmodel().attach_to(base)
                    # gm.gen_sphere(pos=np.average(collideinfo[1],axis=0), radius=0.0051).attach_to(base)

                    # gm.gen_sphere(pos=jaw_center_pos, radius=0.003, rgba=((0, 191 / 255, 1, 0.5))).attach_to(base)



                    # gm.gen_box(extent=[self.blocksize]*3, homomat=rm.homomat_from_posrot(jaw_center_pos), rgba=((0, 191 / 255, 1, 0.5))).attach_to(base)
                    # # gm.gen_arrow(spos=jaw_center_pos, epos=jaw_center_pos -jaw_width * jaw_center_rotmat[:, 0]/2, thickness=0.002,rgba=(1,0,1,0.6)).attach_to(base)
                    # gm.gen_arrow(spos=jaw_center_pos, epos=jaw_center_pos - jaw_width * jaw_center_rotmat[:, 2] / 2,
                    #              thickness=0.002, rgba=(0, 1, 1, 0.6)).attach_to(base)
                    # # gm.gen_arrow(spos=jaw_center_pos -jaw_width * jaw_center_rotmat[:, 0]/2, epos=jaw_center_pos+jaw_width * jaw_center_rotmat[:,0]/2, rgba=(1,0,0,0.6), thickness=0.002).attach_to(base)
                    # gripper.gen_meshmodel(rgba=(1, 0, 0, 0.3)).attach_to(base)
                    pass


                # for mesh in partialmeshCKlist:
                #     # checker = trimesh.base.ray.ray_pyembree.RayMeshIntersector(mesh)
                #     # ray_directions = [jaw_center_rotmat[:,0],-jaw_center_rotmat[:,0]]
                #     # ray_origins = [jaw_center_pos] * 2
                #     # hitinfo = checker.intersects_any(ray_origins=ray_origins, ray_directions=ray_directions)
                #     if any(hitinfo):
                #         briefgrasplist.append(grasp_info)
                #         # gripper.gen_meshmodel(rgba=(1, 0, 0, 0.1)).attach_to(base)
                #         # gm.gen_arrow(spos=jaw_center_pos, epos=jaw_center_pos+0.11 * jaw_center_rotmat[:,0], thickness=0.002).attach_to(base)
                #         # gm.gen_arrow(spos=jaw_center_pos, epos=jaw_center_pos -0.11 * jaw_center_rotmat[:, 0], thickness=0.002).attach_to(base)
                #         break
                # gm.gen_arrow(spos=jaw_center_pos, epos=jaw_center_pos + 0.11 * jaw_center_rotmat[:, 0], rgba=(0.1,0.1,0.1,0.1), thickness=0.002).attach_to(base)
                # gm.gen_arrow(spos=jaw_center_pos, epos=jaw_center_pos - 0.11 * jaw_center_rotmat[:, 0], rgba=(0.1,0.1,0.1,0.1), thickness=0.002).attach_to(base)
                # gripper.gen_meshmodel(rgba=(0.3, 0.3, 0.3, 0.1)).attach_to(base)

        return None

        # gm.GeometricModel(partialmesh).attach_to(target)
        # return target

    def hitray(self, observe_origin = [.0,.0,-.09]):
        checker = trimesh.base.ray.ray_pyembree.RayMeshIntersector(self.mesh)
        # observation = np.array([[0, 0, .900]])
        observation = np.array([observe_origin])
        ray_directions = [point - observation[0] for point in self.points]
        ray_origins = [observation[0]] *len(self.points)
        hitinfo = checker.intersects_id(ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False, max_hits=1,
                            return_locations=True)
        hited_list = []
        for i, point in enumerate(self.points):
            if np.linalg.norm(point - hitinfo[2][i]) <= self.voxel:
                hited_list.append(point)
        hited_list = np.asarray(hited_list)
        # self.observed_mesh = self.voxelizedmodel.points_to_indices(self.hited_list)

        return hited_list

    @property
    def outputTrimesh(self):
        self.newmesh = trimeshWan.Trimesh(vertices=self.vertices, faces=self.faces, face_normals=self.face_normals,
                           vertex_normals=self.vertex_normals)
        return self.newmesh

    def __infoUpdate(self, mesh):
        self.faces = np.asarray(mesh.faces)
        self.vertices = np.asarray(mesh.vertices)
        self.face_normals = np.asarray(mesh.face_normals)
        self.vertex_normals = np.asarray(mesh.vertex_normals)

    def meshTransform(self, rotaxis = np.array([0,0,1]), angle = np.radians(90), translation=np.array([0,0,0])):
        rotmat = rm.rotmat_from_axangle(rotaxis, angle)
        homomate = rm.homomat_from_posrot(translation, rotmat)
        self.mesh.apply_transform(homomate)
        # self.vertices = np.asarray([rotmat.dot(vert) + translation for vert in self.mesh.vertices])
        # self.faces = self.mesh.faces
        # self.face_normals =  np.asarray([rotmat.dot(face_normal) + translation for face_normal in self.mesh.face_normals])
        # self.vertex_normals =  np.asarray([rotmat.dot(vertex_normal) + translation for vertex_normal in self.mesh.vertex_normals])
        # self.mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces, face_normals=self.face_normals,
        #                                vertex_normals=self.vertex_normals)

    def export(self, outputfile, outputname):
        this_dir, this_filename = os.path.split(__file__)
        extention = ".stl"
        if outputfile:
            outputfile = outputfile
        else:
            outputfile = this_dir
        self.mesh.export(outputfile + "/" + outputname + extention)

if __name__ == '__main__':

    base = wd.World(cam_pos=[2.01557, 0.637317, 1.88133], w=960,
                    h=540, lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    this_dir, this_filename = os.path.split(__file__)

    name = "armadillo.stl"
    mesh = TrimeshHu("./3dcnnobj/", name)

    # name = "bo"
    # box = gm.gen_box([.090,.090,.090]).objtrm

    # mesh = TrimeshHu(mesh = box, scale=1)
    # mesh.set_scale((0.001, 0.001, 0.001))
    # mesh.voxelization(45, hollow = False)

    icosphere = gm.gen_sphere(radius=0.2, rgba =[0,0,1,0.1] ,subdivisions=1)
    sample = icosphere.objtrm.vertices
    icosphere.attach_to(base)
    # print(sample)


    import open3d as o3d
    import vision.depth_camera.pcd_data_adapter as vdda
    import freeholdcontactpairs as f
    calib_mat = pickle.load(open("phoxi_calibmat.pkl", "rb"))
    pcd_list = pickle.load(open("pc/bunnypcd.pkl", "rb"))[2]

    pcd_list = rm.homomat_transform_points(calib_mat, pcd_list)
    n_pcd_list = []
    for pcd in pcd_list:
        if pcd[2] > 810 and pcd[2] < 930  and pcd[1]>100 and pcd[1]<270 and pcd[0]>700 and pcd[0]<930:
            n_pcd_list.append(pcd)


    # pcd_list = [pcd for pcd in pcd_list if pcd[2]>785]
    pcd = vdda.nparray_to_o3dpcd(np.asarray(n_pcd_list))
    pcd, ind = pcd.remove_radius_outlier(nb_points=15, radius=5)
    o3d.visualization.draw_geometries([pcd])
    alpha = 20
    mmesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    radii = np.array([0.005, 0.01, 0.02, 0.04])
    # mmesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    mmesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mmesh], mesh_show_back_face=True)
    mmesh_trimesh = vdda.o3dmesh_to_trimesh(mmesh)
    p_obj = cm.CollisionModel(mmesh_trimesh)
    p_obj.set_scale((0.001,0.001,0.001))
    p_obj.set_rgba((0.5, 0.5, 0.5, 1))
    p_obj.attach_to(base)

    mmesh_trimesh.export("partial_bunny.stl")
    partial = f.FreeholdContactpairs("partial_bunny.stl")
    partial.showallFaces()

    base.run()


    mmesh_hu = TrimeshHu(mesh = mmesh_trimesh)
    mmesh_hu.voxelization(3, hollow=True)
    mmesh_hu.get_node_matrix()
    mmesh_hu.get_transform()
    t = cm.CollisionModel(mmesh_hu.outputTrimesh)
    t.set_scale((0.001, 0.001, 0.001))
    t.set_rgba((0, 1, 0, .11))
    t.attach_to(base)
    base.run()


    # pctrim = vdda.o3dmesh_to_trimesh(pcd)
    radii = [0.005, 0.01, 0.02, 0.04]
    cl, ind = pcd.remove_radius_outlier(nb_points=15, radius=0.003)
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    # objpdnp_raw.reparentTo(self._objpdnp)
    a = gm.GeometricModel(pcd)
    a.set_scale((0.001,0.001,0.001))
    a.attach_to(base)
    base.run()


    mesh.meshTransform(rotaxis = np.array([0,0,1]), angle = np.radians(45), translation=np.array([0,0,0]))
    mesh.voxelization(.0045, hollow = True)
    mesh.get_node_matrix()
    mesh.get_transform()
    # mesh.show_balls()
    # mesh.show_hited_balls(base)
    mesh.export(this_dir,"box_vox")
    c = cm.CollisionModel(mesh.outputTrimesh)
    # c.set_scale((0.001, 0.001, 0.001))
    c.set_rgba((0,1,0,.11))
    c.attach_to(base)


    objNode = [None]
    voxelNode = [None]
    observeNode = [None]


    def update(textNode, objNode, voxelNode, observeNode, count, task):
        if observeNode[0] is not None:
            observeNode[0].detachNode()
        observeNode[0] = NodePath("observe")
        mesh.show_hited_balls(observe_origin=sample[count[0]],target = observeNode[0])
        gm.gen_sphere(sample[count[0]]).attach_to(observeNode[0])
        observeNode[0].reparent_to(base.render)
        count[0]+=1

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

    cam_view_text = OnscreenText(
        text="Camera View: ",
        fg=(0, 0, 0, 1),
        pos=(1.15, 0.9),
        align=TextNode.ALeft)
    testNode = [None, None, None]
    count = [0]
    taskMgr.doMethodLater(1,update,  "addobject", extraArgs=[testNode, objNode, voxelNode, observeNode, count], appendTask=True)




    # print(b)
    base.run()


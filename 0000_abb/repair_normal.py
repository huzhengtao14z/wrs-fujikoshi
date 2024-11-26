import copy
import math
import visualization.panda.world as wd
import modeling.collision_model as cm
import humath as hm
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode
import os
import basis.trimesh as trimeshWan
import trimesh as trimesh


if __name__ == '__main__':
    base = wd.World(cam_pos=[2.01557, 0.637317, 1.88133], w=960,
                    h=540, lookat_pos=[0, 0, 0])
    # gm.gen_frame().attach_to(base)
    this_dir, this_filename = os.path.split(__file__)
    # filepath = 'kit_model'
    filepath = 'object'
    obj_name_list = os.listdir(filepath)
    filepath_output = '.'
    for obj_name in obj_name_list:
        mesh = trimesh.load_mesh(filepath+'/'+obj_name)
        obj_name = obj_name.split(".")[0]+".stl"
        mesh.fix_normals()
        mesh.export(filepath_output+'/'+obj_name)
    obj = cm.CollisionModel(filepath_output+'/'+obj_name_list[1].split(".")[0]+".stl")
    obj.set_rgba([1, 1, 0, 0.5])
    obj.attach_to(base)
    base.run()



    def update(textNode, count, task):

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
    taskMgr.doMethodLater(0.01, update, "addobject", extraArgs=[testNode, count],
                          appendTask=True)

    base.run()
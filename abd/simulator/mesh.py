# import meshio
import warp as wp
import numpy as np
import igl
from pxr import UsdGeom, Usd
import pyHouGeoIO

class RawMeshFromFile:
    '''
    load mesh from file. 
    Supported formats: obj, usd, bgeo
    return format: (#V x 3), (#F x 3)
    '''
    @staticmethod
    def load_usd_mesh(filename, usd_path):
        asset_stage = Usd.Stage.Open(filename)
        mesh_geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath(usd_path))

        points = np.array(mesh_geom.GetPointsAttr().Get())
        indices = np.array(mesh_geom.GetFaceVertexIndicesAttr().Get())

        return points, indices

    @staticmethod
    def load_bgeo_triangle_mesh(filename, path):
        mesh = pyHouGeoIO.HTriangleMesh()
        
        pyHouGeoIO.ImportHouGeo(path + filename, mesh)
        # points, indices = wp.load_bgeo_mesh(filename, path)
        # points = np.zeros((0, 3), dtype = np.float32)
        indices = np.zeros((0, 3), dtype = int)
        
        points = mesh.GetPointAttributeMatXf("P")
        indices = mesh.GetTriangleTopology().T
        return points, indices

    def __init__(self, file = "assets/stanford-bunny.obj", folder = "", usd_path = "/Cube"):
        self.V = np.zeros((0, 3))
        self.F = np.zeros((0, 3), dtype = int)
        vertices, indices = self.V, self.F
        if file in ["box", "sphere"]:
            '''predefined shapes'''
            return 
        elif file.endswith((".usd", ".usda", "usdz")):
            '''load usd'''
            vertices, indices = self.load_usd_mesh(folder + file, usd_path)
        elif file.endswith((".bgeo")):
            vertices, indices = self.load_bgeo_triangle_mesh(file, folder)
        else:
            '''load obj'''
            vertices, indices = igl.read_triangle_mesh(folder + file)
            indices = np.array(indices, dtype = int).reshape(-1, 3)
            # vertices, _, _, indices, _, _ = igl.read_obj("my_model.obj")
        self.V, self.F = vertices, indices

if __name__ == "__main__":
    files = ["box.bgeo", "box.obj", "cube.usda"]
    for file in files:
        raw_mesh = RawMeshFromFile(file, "assets/")
        print(file, raw_mesh.V.shape, raw_mesh.F.shape)
from mesh import RawMeshFromFile
import numpy as np
import json
from typing import List
class KineticMesh(RawMeshFromFile):

    @staticmethod
    def states_default():
        '''
        change these two methods to define mesh path, states and their default values
        '''

        return {
            "x": np.zeros(3),
            "q": np.quaternion(1, 0, 0, 0),
            "v": np.zeros(3),
            "omega": np.zeros(3),
            "mass": 1.0,
            "I0": np.identity(3)
        }

    @staticmethod
    def mesh_path_default():
        return {
            "file": "assets/stanford-bunny.obj",
            "folder": "",
            "usd_path": "/Cube"
        }


    def __init__(self, obj_json):
        mesh_path = self.mesh_path_default()
        for key, value in mesh_path.items():
            if key in obj_json:
                mesh_path[key] = obj_json[key]

        file, folder, usd_path = mesh_path
        super().__init__(file, folder, usd_path)


        states = self.states_default()
        for key, value in states.items():
            if key in obj_json:
                value = np.quaternion(obj_json[key]) if key == "q" else np.array(obj_json[key])
                states[key] = value
            setattr(self, key, value)
        
        for key in states.keys():
            assert(hasattr(self, key))

class Scene: 
    def __init__(self, scene_config_file = "scenes/case1.json"):
        self.kinetic_objects: List[KineticMesh] = []
        with open(scene_config_file) as f:
            scene_config = json.load(f)

        for obj in scene_config["objects"]:
            self.kinetic_objects.append(KineticMesh(obj))
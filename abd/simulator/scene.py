from .mesh import RawMeshFromFile
import numpy as np
import json
from typing import List, TypeVar, Generic
T = TypeVar('T')
class KineticMesh(RawMeshFromFile):

    @classmethod
    def states_default(cls):
        '''
        change these two methods to define mesh path, states and their default values
        '''

        return {
            "x": np.zeros(3),
            "q": np.array([1, 0, 0, 0]),
            "v": np.zeros(3),
            "omega": np.zeros(3),
            "mass": 1.0,
            "I0": np.identity(3)
        }

    @classmethod
    def mesh_path_default(cls):
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

        file, folder, usd_path = mesh_path.values()
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
    def __init__(self, member_type = KineticMesh, scene_config_file = "scenes/case1.json"):
        print(scene_config_file)
        self.kinetic_objects: List[member_type] = []
        with open(scene_config_file) as f:
            scene_config = json.load(f)

        for obj in scene_config["objects"]:
            self.kinetic_objects.append(member_type(obj))


if __name__ == "__main__":
    scene = Scene(KineticMesh, scene_config_file = "scenes/case1.json")
    
    print(scene.kinetic_objects[0].V.shape)
    
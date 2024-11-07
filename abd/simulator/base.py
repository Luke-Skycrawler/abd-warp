from .scene import KineticMesh, Scene
import json
import os
import numpy as np
import pyHouGeoIO as hou
import polyscope as ps

class BaseSimulator:

    @staticmethod
    def top_module_path():
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


    @classmethod
    def simulator_args(cls):
        '''
        change this method to define simulator arguments and default values (values can be overwritten by config file)
        '''
        
        return {
            "frame_in": 0,
            "frame_out": 100,
            "save_sequence": False,
            "input_seq_folder": os.path.join(BaseSimulator.top_module_path(), "bgeo/"),
            "output_seq_folder": os.path.join(BaseSimulator.top_module_path(), "bgeo/"),
            "verbose": 0
        }


    def __init__(self, config_file = "config.json", member_type = KineticMesh):
        sim_args = self.simulator_args()

        with open(config_file) as f:
            config = json.load(f)

        for key, value in sim_args.items():
            if key in config:
                value = config[key]
                setattr(self, key, value)
            else:
                setattr(self, key, value)

        scene_config = config["scene"]
        self.scene = Scene(member_type=member_type, scene_config_file = scene_config)

    def reset(self): 
        pass
    
    @classmethod
    def bgeo_object_attributes(cls):
        '''
        change this method to define bgeo attributes and their default values
        '''
        
        return {
            "A": "MatXf",
            "b": "VecXf"
        }
    


    def load_bgeo(self, frame = 0):
        '''
        load bgeo file to initialize the scene
        '''
        for key, type in self.bgeo_object_attributes().items():
            setattr(self, key, )

    def export_frame(self):
        '''
        export current frame in reentrant bgeo format
        '''
        
        pass

    # supporting functions 
    def gather(self, fieldname):
        return np.array([getattr(ko, fieldname) for ko in self.scene.kinetic_objects])
        
    
    
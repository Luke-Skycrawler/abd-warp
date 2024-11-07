#### Code Structure

simulator/ 

*deals with io, all members in numpy format*.
- mesh.py: 

    `RawMeshFromFile`: handles mesh io, defines np.ndarray V, F.
- scene.py: 

    `KineticMesh(RawMeshFromFile)`(not used in abd): defines rigid body dofs in np.

    `Scene`: takes a member type `T` (e.g. KineticMesh) and translates json config to `self.kinetic_objects: List[T]`. The json attribute name should match the attributes defined in `T.states_default()` static method. 
- base.py:

    `BaseSimulator`: defines simulator basic simulator arguments, including specifying start/end frames, save sequence folder, verbose levels. Loads the scene in `self.scene: Scene`.

abd/

- affine_body.py:
    
    `AffineMesh(KineticMesh)`: load json config to numpy structure, defines affine dofs. Further computes `self.edges`. Can bridge to `WarpMesh` if assigned with an `id` (but does not include dynamics dof, which is sent to `AfineBodyStates` by `BaseSimulator.gather()`). 

    `(@wp.struct)WarpMesh`: warp structure that stores a surface mesh.

    `(@wp.struct)AffineBodyStates`: warp Structure of Arrays that stores affine simulation states.

    
    

    
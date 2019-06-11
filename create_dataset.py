import ai2thor.controller
import numpy as np
from collections import namedtuple
from keras.applications import resnet50
import h5py
import json
import time
names = ["FloorPlan1", "FloorPlan201", "FloorPlan301", "FloorPlan401"]
grid_size = 0.5

actions = ["MoveAhead", "RotateRight", "RotateLeft", "MoveBack", "LookUp", "LookDown"]
rotation_possible_inplace = 4
ACTION_SIZE = len(actions)

def equal(s1, s2):
    if s1.pos == s2.pos:
            if s1.rot == s2.rot:
                return True
    return False

def search_namedtuple(states, search):
    for state in states:
        if equal(state, search):
            return state
    return None

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == '__main__':

    controller = ai2thor.controller.Controller()
    controller.start(player_screen_width = 400, player_screen_height = 300) 

    for name in names:
        h5_file = h5py.File("data/" + name + '.h5', 'w')

        #Use resnet from Keras to compute features
        resnet_trained = resnet50.ResNet50(include_top=False, weights = 'imagenet', pooling='avg', input_shape = (300, 400, 3))
        # Freeze all layers
        for layer in resnet_trained.layers:
            layer.trainable = False





        # Reset the environnment
        controller.reset(name)

        # gridSize specifies the coarseness of the grid that the agent navigates on
        state = controller.step(dict(action='Initialize', gridSize=grid_size, renderObjectImage=True))
        
        reachable_pos = controller.step(dict(action='GetReachablePositions', gridSize=grid_size)).metadata['reachablePositions']

        # Construct all possible states
        states = []
        obss = []
        StateStruct = namedtuple("StateStruct", "id pos rot obs feat bbox")
        
        id = 0
        for pos in reachable_pos:
            state = controller.step(dict(action='Teleport', **pos))
            for a in range(rotation_possible_inplace):
                state = controller.step(dict(action="RotateLeft"))
                # Normal/Up/Down view
                for i in range(3):
                    # Up view
                    if i == 1:
                        state = controller.step(dict(action="LookUp"))
                        state.metadata['agent']['rotation']['z'] = state.metadata['agent']['cameraHorizon']
                    # Down view
                    elif i == 2:
                        state = controller.step(dict(action="LookDown"))
                        state.metadata['agent']['rotation']['z'] = state.metadata['agent']['cameraHorizon']
                    obs_process = resnet50.preprocess_input(state.frame)
                    obs_process = obs_process[np.newaxis,...]
                    
                    feature = resnet_trained.predict(obs_process)
                    state_struct = StateStruct(id, 
                                            state.metadata['agent']['position'],
                                            state.metadata['agent']['rotation'],
                                            obs=state.frame, 
                                            feat=feature, 
                                            bbox=json.dumps(state.instance_detections2D, cls=NumpyEncoder))
                    time.sleep(0.4)
                    if search_namedtuple(states, state_struct):
                        print("Already exists")
                        exit()

                    states.append(state_struct)
                    id = id + 1

                    # Reset camera
                    if i == 1:
                        state = controller.step(dict(action="LookDown"))
                    elif i == 2:
                        state = controller.step(dict(action="LookUp"))

        # Create action-state graph
        # Each state will be at 1 position and 4 rotation
        num_states = len(states)
        graph = np.full((num_states, ACTION_SIZE), -1)

        # Populate graph
        for state in states:
            for i in range(len(actions)):
                controller.step(dict(action='TeleportFull', **state.pos, rotation=state.rot['y']))
                state_controller = controller.step(dict(action=actions[i]))
                # Convert to search
                state_controller_named = StateStruct(-1, 
                                                     state_controller.metadata['agent']['position'],
                                                     state_controller.metadata['agent']['rotation'], 
                                                     obs = None, 
                                                     feat = None,
                                                     bbox = None)

                if not equal(state, state_controller_named):
                    found = search_namedtuple(states, state_controller_named)
                    if found is None:
                        print("Error, state not found")
                        exit()
                    graph[state.id][i] = found.id

        # Save it to h5 file
        h5_file.create_dataset('resnet_feature', data = [s.feat for s in states])
        h5_file.create_dataset('observation', data = [s.obs for s in states])
        h5_file.create_dataset('location', data = [list(s.pos.values()) for s in states])
        h5_file.create_dataset('rotation', data = [list(s.rot.values()) for s in states])
        h5_file.create_dataset('bbox', data = [s.bbox.encode("ascii", "ignore") for s in states])
        h5_file.create_dataset('graph', data = graph)

        shortest_path_distance = np.ones((num_states,num_states))
        h5_file.create_dataset('shortest_path_distance', data = shortest_path_distance)
        h5_file.close()

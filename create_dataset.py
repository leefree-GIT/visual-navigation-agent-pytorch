import argparse
import json
import os
import re
from collections import namedtuple

import ai2thor.controller
import h5py
import numpy as np
import spacy
from keras.applications import resnet50
from PIL import Image
from tqdm import tqdm

names = ["FloorPlan1", "FloorPlan2", "FloorPlan201", "FloorPlan202",
         "FloorPlan301", "FloorPlan302", "FloorPlan401", "FloorPlan402"]
grid_size = 0.5

actions = ["MoveAhead", "RotateRight", "RotateLeft",
           "MoveBack", "LookUp", "LookDown", "MoveRight", "MoveLeft"]
rotation_possible_inplace = 4
ACTION_SIZE = len(actions)


def equal(s1, s2):
    if s1.pos == s2.pos:
        if s1.rot == s2.rot:
            return True
    return False


def search_namedtuple(list_states, search_state):
    for s in list_states:
        if equal(s, search_state):
            return s
    return None


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == '__main__':
    argparse.ArgumentParser(description="")
    parser = argparse.ArgumentParser(description='Dataset creation.')
    parser.add_argument('--eval', action='store_true')
    args = vars(parser.parse_args())
    controller = ai2thor.controller.Controller()

    w, h = 400, 300
    controller.start(player_screen_width=w, player_screen_height=h)

    # Use resnet from Keras to compute features
    resnet_trained = resnet50.ResNet50(
        include_top=False, weights='imagenet', pooling='avg', input_shape=(h, w, 3))
    # Freeze all layers
    for layer in resnet_trained.layers:
        layer.trainable = False

    i = 0
    pbar_names = tqdm(names)

    # Use scapy to extract vector from word embeddings
    nlp = spacy.load('en_core_web_lg')  # Use en_core_web_lg for more words

    # Use glob to list object image
    import glob
    object_id = 0
    object_ids = {}
    object_feature = []
    object_vector = []
    # List all jpg files in data/objects/
    for filepath in glob.glob('data/objects/*.jpg'):

        # Resize image to be the same as observation (300x400)
        frame = Image.open(filepath)
        frame = frame.resize((w, h))
        frame = np.asarray(frame, dtype="int32")

        # Use resnet to extract object features
        obj_process = resnet50.preprocess_input(frame)
        obj_process = obj_process[np.newaxis, ...]
        feature = resnet_trained.predict(obj_process)

        filename = os.path.splitext(os.path.basename(filepath))[0]
        object_ids[filename] = object_id
        object_feature.append(feature)

        # Usee scapy to extract word embedding vector
        word_vec = nlp(filename.lower())

        # If words don't exist in dataset
        # cut them using uppercase letter (SoapBottle -> [Soap, Bottle])
        if word_vec.vector_norm == 0:
            word_split = re.findall('[A-Z][^A-Z]*', filename)
            for word in word_split:
                word_vec = nlp(word.lower())
                if word_vec.vector_norm == 0:
                    break
    for name in pbar_names:
        pbar_names.set_description("%s" % name)
        if args['eval']:
            if not os.path.exists("data_eval/"):
                os.makedirs("data_eval/")
            h5_file = h5py.File("data_eval/" + name + '.h5', 'a')
        else:
            if not os.path.exists("data/"):
                os.makedirs("data/")
            h5_file = h5py.File("data/" + name + '.h5', 'a')

            norm_word_vec = word_vec.vector / word_vec.vector_norm
            object_vector.append(norm_word_vec)

        if 'object_feature' in h5_file.keys():
            del h5_file['object_feature']
        h5_file.create_dataset(
            'object_feature', data=object_feature)

        if 'object_vector' in h5_file.keys():
            del h5_file['object_vector']
        h5_file.create_dataset(
            'object_vector', data=object_vector)

        # Reset the environnment
        controller.reset(name)
        if not args['eval']:
            controller.step(dict(action='InitialRandomSpawn',
                                 randomSeed=0, forceVisible=True, maxNumRepeats=5))
        else:
            controller.step(dict(action='InitialRandomSpawn',
                                 randomSeed=100, forceVisible=True, maxNumRepeats=5))

        # gridSize specifies the coarseness of the grid that the agent navigates on
        state = controller.step(
            dict(action='Initialize', gridSize=grid_size, renderObjectImage=True))

        reachable_pos = controller.step(dict(
            action='GetReachablePositions', gridSize=grid_size)).metadata['reachablePositions']

        # Construct all possible states
        states = []
        obss = []
        StateStruct = namedtuple("StateStruct", "id pos rot obs feat bbox")

        idx = 0
        if 'resnet_feature' not in h5_file.keys() or \
           'observation' not in h5_file.keys() or \
           'location' not in h5_file.keys() or \
           'rotation' not in h5_file.keys() or \
           'bbox' not in h5_file.keys():
            for pos in tqdm(reachable_pos, desc="Feature extraction"):
                state = controller.step(dict(action='Teleport', **pos))
                # Normal/Up/Down view
                for i in range(3):
                    # Up view
                    if i == 1:
                        state = controller.step(dict(action="LookUp"))
                    # Down view
                    elif i == 2:
                        state = controller.step(dict(action="LookDown"))
                    # Rotate
                    for a in range(rotation_possible_inplace):
                        state = controller.step(dict(action="RotateLeft"))
                        state.metadata['agent']['rotation']['z'] = state.metadata['agent']['cameraHorizon']
                        obs_process = resnet50.preprocess_input(state.frame)
                        obs_process = obs_process[np.newaxis, ...]

                        feature = resnet_trained.predict(obs_process)

                        state_struct = StateStruct(
                            idx,
                            state.metadata['agent']['position'],
                            state.metadata['agent']['rotation'],
                            obs=state.frame,
                            feat=feature,
                            bbox=json.dumps(state.instance_detections2D, cls=NumpyEncoder))

                        if search_namedtuple(states, state_struct):
                            print("Already exists")
                            exit()

                        states.append(state_struct)
                        idx = idx + 1

                    # Reset camera
                    if i == 1:
                        state = controller.step(dict(action="LookDown"))
                    elif i == 2:
                        state = controller.step(dict(action="LookUp"))

            # Save it to h5 file
            if 'resnet_feature' in h5_file.keys():
                del h5_file['resnet_feature']
            h5_file.create_dataset(
                'resnet_feature', data=[s.feat for s in states])

            if 'observation' in h5_file.keys():
                del h5_file['observation']
            h5_file.create_dataset(
                'observation', data=[s.obs for s in states])

            if 'location' in h5_file.keys():
                del h5_file['location']
            h5_file.create_dataset(
                'location', data=[list(s.pos.values()) for s in states])

            if 'rotation' in h5_file.keys():
                del h5_file['rotation']
            h5_file.create_dataset(
                'rotation', data=[list(s.rot.values()) for s in states])

            if 'bbox' in h5_file.keys():
                del h5_file['bbox']
            h5_file.create_dataset(
                'bbox', data=[s.bbox.encode("ascii", "ignore") for s in states])

        # Create action-state graph
        # Each state will be at 1 position and 4 rotation
        num_states = len(states)
        graph = np.full((num_states, ACTION_SIZE), -1)

        # Speed improvement
        state = controller.step(
            dict(action='Initialize', gridSize=grid_size, renderObjectImage=False))
        # Populate graph
        if 'graph' not in h5_file.keys():
            for state in tqdm(states, desc="Graph construction"):
                for i, a in enumerate(actions):
                    controller.step(dict(action='TeleportFull', **state.pos,
                                         rotation=state.rot['y'], horizon=state.rot['z']))
                    state_controller = controller.step(dict(action=a))
                    state_controller.metadata['agent']['rotation']['z'] = state_controller.metadata['agent']['cameraHorizon']
                    # Convert to search
                    state_controller_named = StateStruct(-1,
                                                         state_controller.metadata['agent']['position'],
                                                         state_controller.metadata['agent']['rotation'],
                                                         obs=None,
                                                         feat=None,
                                                         bbox=None)

                    if not equal(state, state_controller_named) and not round(state_controller.metadata['agent']['cameraHorizon']) == 60:
                        found = search_namedtuple(
                            states, state_controller_named)
                        if found is None:
                            # print(state_controller_named)
                            # print("Error, state not found")
                            # exit()
                            continue
                        graph[state.id][i] = found.id

            h5_file.create_dataset(
                'graph', data=graph)

        h5_file.attrs["object_ids"] = np.string_(json.dumps(object_ids))

        shortest_path_distance = np.ones((num_states, num_states))
        if 'shortest_path_distance' not in h5_file.keys():
            h5_file.create_dataset('shortest_path_distance',
                                   data=shortest_path_distance)
        h5_file.close()

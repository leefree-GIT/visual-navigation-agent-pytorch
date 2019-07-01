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
StateStruct = namedtuple("StateStruct", "id pos rot obs feat bbox obj_visible")

# Extracted from unity/Assets/Scripts/SimObjType.cs
OBJECT_IDS = {
    "Undefined": 0,
    "Apple": 1,
    "AppleSliced": 2,
    "Tomato": 3,
    "TomatoSliced": 4,
    "Bread": 5,
    "BreadSliced": 6,
    "Sink": 7,
    "Pot": 8,
    "Pan": 9,
    "Knife": 10,
    "Fork": 11,
    "Spoon": 12,
    "Bowl": 13,
    "Toaster": 14,
    "CoffeeMachine": 15,
    "Microwave": 16,
    "StoveBurner": 17,
    "Fridge": 18,
    "Cabinet": 19,
    "Egg": 20,
    "Chair": 21,
    "Lettuce": 22,
    "Potato": 23,
    "Mug": 24,
    "Plate": 25,
    "TableTop": 26,
    "CounterTop": 27,
    "GarbageCan": 28,
    "Omelette": 29,
    "EggShell": 30,
    "EggCracked": 31,
    "StoveKnob": 32,
    "Container": 33,
    "Cup": 34,
    "ButterKnife": 35,
    "PotatoSliced": 36,
    "MugFilled": 37,
    "BowlFilled": 38,
    "Statue": 39,
    "LettuceSliced": 40,
    "ContainerFull": 41,
    "BowlDirty": 42,
    "Sandwich": 43,
    "Television": 44,
    "HousePlant": 45,
    "TissueBox": 46,
    "VacuumCleaner": 47,
    "Painting": 48,
    "WateringCan": 49,
    "Laptop": 50,
    "RemoteControl": 51,
    "Box": 52,
    "Newspaper": 53,
    "TissueBoxEmpty": 54,
    "PaintingHanger": 55,
    "KeyChain": 56,
    "Dirt": 57,
    "CellPhone": 58,
    "CreditCard": 59,
    "Cloth": 60,
    "Candle": 61,
    "Toilet": 62,
    "Plunger": 63,
    "Bathtub": 64,
    "ToiletPaper": 65,
    "ToiletPaperHanger": 66,
    "SoapBottle": 67,
    "SoapBottleFilled": 68,
    "SoapBar": 69,
    "ShowerDoor": 70,
    "SprayBottle": 71,
    "ScrubBrush": 72,
    "ToiletPaperRoll": 73,
    "Lamp": 74,
    "LightSwitch": 75,
    "Bed": 76,
    "Book": 77,
    "AlarmClock": 78,
    "SportsEquipment": 79,
    "Pen": 80,
    "Pencil": 81,
    "Blinds": 82,
    "Mirror": 83,
    "TowelHolder": 84,
    "Towel": 85,
    "Watch": 86,
    "MiscTableObject": 87,
    "ArmChair": 88,
    "BaseballBat": 89,
    "BasketBall": 90,
    "Faucet": 91,
    "Boots": 92,
    "Glassbottle": 93,
    "DishSponge": 94,
    "Drawer": 95,
    "FloorLamp": 96,
    "Kettle": 97,
    "LaundryHamper": 98,
    "LaundryHamperLid": 99,
    "Lighter": 100,
    "Ottoman": 101,
    "PaintingSmall": 102,
    "PaintingMedium": 103,
    "PaintingLarge": 104,
    "PaintingHangerSmall": 105,
    "PaintingHangerMedium": 106,
    "PaintingHangerLarge": 107,
    "PanLid": 108,
    "PaperTowelRoll": 109,
    "PepperShaker": 110,
    "PotLid": 111,
    "SaltShaker": 112,
    "Safe": 113,
    "SmallMirror": 114,
    "Sofa": 115,
    "SoapContainer": 116,
    "Spatula": 117,
    "TeddyBear": 118,
    "TennisRacket": 119,
    "Tissue": 120,
    "Vase": 121,
    "WallMirror": 122,
    "MassObjectSpawner": 123,
    "MassScale": 124,
    "Footstool": 125,
    "Shelf": 126,
    "Dresser": 127,
    "Desk": 128,
    "NightStand": 129,
    "Pillow": 130,
    "Bench": 131,
    "Cart": 132,
    "ShowerGlass": 133,
    "DeskLamp": 134,
    "Window": 135,
    "BathtubBasin": 136,
    "SinkBasin": 137,
    "CD": 138,
    "Curtains": 139,
    "Poster": 140,
    "HandTowel": 141,
    "HandTowelHolder": 142,
    "Ladle": 143,
    "WineBottle": 144,
    "ShowerCurtain": 145,
    "ShowerHead": 146

}


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


def create_states(h5_file, resnet_trained, controller, name, args):
    # Reset the environnment
    controller.reset(name)
    if args['eval']:
        controller.step(dict(action='InitialRandomSpawn',
                             randomSeed=100, forceVisible=True, maxNumRepeats=5))

    # gridSize specifies the coarseness of the grid that the agent navigates on
    state = controller.step(
        dict(action='Initialize', gridSize=grid_size, renderObjectImage=True))

    reachable_pos = controller.step(dict(
        action='GetReachablePositions', gridSize=grid_size)).metadata['reachablePositions']

    states = []
    obss = []
    idx = 0
    # Does not redo if already existing
    if args['force'] or \
        'resnet_feature' not in h5_file.keys() or \
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

                    # Extract resnet feature from observation
                    feature = resnet_trained.predict(obs_process)

                    # Store visible objects from the agent (visible = 1m away from the agent)
                    obj_visible = [obj['objectId']
                                   for obj in state.metadata['objects'] if obj['visible']]
                    state_struct = StateStruct(
                        idx,
                        state.metadata['agent']['position'],
                        state.metadata['agent']['rotation'],
                        obs=state.frame,
                        feat=feature,
                        bbox=json.dumps(
                            state.instance_detections2D, cls=NumpyEncoder),
                        obj_visible=json.dumps(obj_visible))

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

        if 'object_visibility' in h5_file.keys():
            del h5_file['object_visibility']
        h5_file.create_dataset(
            'object_visibility', data=[s.obj_visible.encode("ascii", "ignore") for s in states])
    return states


def create_graph(h5_file, states, controller, args):
    num_states = len(states)
    graph = np.full((num_states, ACTION_SIZE), -1)
    # Speed improvement
    state = controller.step(
        dict(action='Initialize', gridSize=grid_size, renderObjectImage=False))
    # Populate graph
    if args['force'] or 'graph' not in h5_file.keys():
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
                                                     bbox=None,
                                                     obj_visible=None)

                if not equal(state, state_controller_named) and not round(state_controller.metadata['agent']['cameraHorizon']) == 60:
                    found = search_namedtuple(
                        states, state_controller_named)
                    if found is None:
                        # print(state_controller_named)
                        print("Error, state not found")
                        # exit()
                        continue
                    graph[state.id][i] = found.id
        if 'graph' in h5_file.keys():
            del h5_file['graph']

        h5_file.create_dataset(
            'graph', data=graph)
        return graph


def write_object_feature(h5_file, object_feature, object_vector):
    # Write object_feature (resnet features)
    if 'object_feature' in h5_file.keys():
        del h5_file['object_feature']
    h5_file.create_dataset(
        'object_feature', data=object_feature)

    # Write object_vector (word embedding features)
    if 'object_vector' in h5_file.keys():
        del h5_file['object_vector']
    h5_file.create_dataset(
        'object_vector', data=object_vector)

    h5_file.attrs["object_ids"] = np.string_(json.dumps(OBJECT_IDS))


def extract_word_emb_vector(nlp, word_name):
    # Usee scapy to extract word embedding vector
    word_vec = nlp(word_name.lower())

    # If words don't exist in dataset
    # cut them using uppercase letter (SoapBottle -> Soap Bottle)
    if word_vec.vector_norm == 0:
        word = re.sub(r"(?<=\w)([A-Z])", r" \1", word_name)
        word_vec = nlp(word.lower())

        # If no embedding found try to cut word to find embedding (SoapBottle -> [Soap, Bottle])
        if word_vec.vector_norm == 0:
            word_split = re.findall('[A-Z][^A-Z]*', word)
            for word in word_split:
                word_vec = nlp(word.lower())
                if word_vec.has_vector:
                    break
            if word_vec.vector_norm == 0:
                print('ERROR: %s not found' % word_name)
                return None
    norm_word_vec = word_vec.vector / word_vec.vector_norm  # Normalize vector size
    return norm_word_vec


def extract_object_feature(resnet_trained, h, w):
    # Use scapy to extract vector from word embeddings
    nlp = spacy.load('en_core_web_lg')  # Use en_core_web_lg for more words

    # Use glob to list object image
    import glob

    # 2048 is the resnet feature size
    object_feature = np.zeros((len(OBJECT_IDS), 2048), dtype=np.float32)
    # 300 is the word embeddings feature size
    object_vector = np.zeros((len(OBJECT_IDS), 300), dtype=np.float32)
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
        object_feature[OBJECT_IDS[filename]] = feature

    for object_name, object_id in OBJECT_IDS.items():
        norm_word_vec = extract_word_emb_vector(nlp, object_name)
        if norm_word_vec is None:
            print(object_name)
        object_vector[object_id] = norm_word_vec

    return object_feature, object_vector


def create_shortest_path(h5_file, states, graph):
    # Usee network to compute shortest path
    import networkx as nx
    num_states = len(states)
    G = nx.Graph()
    shortest_dist_graph = np.full((num_states, num_states), -1)

    for state in states:
        G.add_node(state.id)
    for state in states:
        for i, a in enumerate(actions):
            if graph[state.id][i] != -1:
                G.add_edge(state.id, graph[state.id][i])
    shortest_path = nx.shortest_path(G)
    for state_id_src in range(num_states):
        for state_id_dst in range(num_states):
            shortest_dist_graph[state_id_src][state_id_dst] = len(
                shortest_path[state_id_src][state_id_dst]) - 1

    if 'shortest_path_distance' in h5_file.keys():
        del h5_file['shortest_path_distance']
    h5_file.create_dataset('shortest_path_distance',
                           data=shortest_dist_graph)


def main():
    argparse.ArgumentParser(description="")
    parser = argparse.ArgumentParser(description='Dataset creation.')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--force', action='store_true')
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

    object_feature, object_vector = extract_object_feature(
        resnet_trained, h, w)

    pbar_names = tqdm(names)

    for name in pbar_names:
        pbar_names.set_description("%s" % name)

        # Eval dataset
        if args['eval']:
            if not os.path.exists("data_eval/"):
                os.makedirs("data_eval/")
            h5_file = h5py.File("data_eval/" + name + '.h5', 'a')
        else:
            if not os.path.exists("data/"):
                os.makedirs("data/")
            h5_file = h5py.File("data/" + name + '.h5', 'a')

        write_object_feature(h5_file,
                             object_feature, object_vector)

        # Construct all possible states
        states = create_states(h5_file, resnet_trained,
                               controller, name, args)

        # Create action-state graph
        graph = create_graph(h5_file, states, controller, args)

        # Create shortest path from all state
        create_shortest_path(h5_file, states, graph)

        h5_file.close()


if __name__ == '__main__':
    main()

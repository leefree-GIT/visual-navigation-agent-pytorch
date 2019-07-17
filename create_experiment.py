import argparse
import json

import h5py
import numpy as np

names = []
SCENES = [0, 200, 300, 400]
TRAIN_SPLIT = (1, 22)
TEST_SPLIT = (21, 26)


KITCHEN_OBJECT_CLASS_LIST = [
    "Toaster",
    "Microwave",
    "Fridge",
    "CoffeeMachine",
    "GarbageCan",
    "Bowl",
]

LIVING_ROOM_OBJECT_CLASS_LIST = [
    "Pillow",
    "Laptop",
    "Television",
    "GarbageCan",
    "Bowl",
]

BEDROOM_OBJECT_CLASS_LIST = ["HousePlant", "Lamp", "Book", "AlarmClock"]


BATHROOM_OBJECT_CLASS_LIST = [
    "Sink", "ToiletPaper", "SoapBottle", "LightSwitch"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create param.json experiment file')
    parser.add_argument('--train_range', nargs=2, default=TRAIN_SPLIT,
                        help='train scene range Ex : 1 12')

    parser.add_argument('--eval_range', nargs=2, default=TEST_SPLIT,
                        help='train scene range Ex : 22 27')
    parser.add_argument('--method', type=str, default="word2vec",
                        help='Method to use Ex : word2vec')
    parser.add_argument('--reward', type=str, default="soft_goal",
                        help='Method to use Ex : soft_goal')

    args = vars(parser.parse_args())
    str_range = list(args["train_range"])
    for i, s in enumerate(str_range):
        str_range[i] = int(s)
    args["train_range"] = str_range

    str_range = list(args["eval_range"])
    for i, s in enumerate(str_range):
        str_range[i] = int(s)
    args["eval_range"] = str_range
    data = {}

    scene_tasks = [KITCHEN_OBJECT_CLASS_LIST, LIVING_ROOM_OBJECT_CLASS_LIST,
                   BEDROOM_OBJECT_CLASS_LIST, BATHROOM_OBJECT_CLASS_LIST]

    training = {}
    for idx_scene, scene in enumerate(SCENES):
        for t in range(*args['train_range']):
            name = "FloorPlan" + str(scene + t)
            f = h5py.File("data/"+name+".h5")
            # Use h5py object available
            obj_available = json.loads(f.attrs["task_present"])
            obj_available = np.array(obj_available)
            obj_available_mask = [False for i in obj_available]
            obj_available_mask = np.array(obj_available_mask)

            object_visibility = [json.loads(j) for j in
                                 f['object_visibility']]
            for obj_visible in object_visibility:
                for objectId in obj_visible:
                    obj = objectId.split('|')
                    for obj_idx, curr_obj in enumerate(obj_available):
                        if obj[0] == curr_obj:
                            obj_available_mask[obj_idx] = True

            training[name] = [{"object": obj}
                              for obj in obj_available[obj_available_mask == True]]

    evaluation = {}
    for idx_scene, scene in enumerate(SCENES):
        for t in range(*args['eval_range']):
            name = "FloorPlan" + str(scene + t)
            # Use h5py object available
            f = h5py.File("data/"+name+".h5")
            obj_available = json.loads(f.attrs["task_present"])
            obj_available = np.array(obj_available)
            obj_available_mask = [False for i in obj_available]
            obj_available_mask = np.array(obj_available_mask)

            object_visibility = [json.loads(j) for j in
                                 f['object_visibility']]
            for obj_visible in object_visibility:
                for objectId in obj_visible:
                    obj = objectId.split('|')
                    for obj_idx, curr_obj in enumerate(obj_available):
                        if obj[0] == curr_obj:
                            obj_available_mask[obj_idx] = True
            evaluation[name] = [
                {"object": obj} for obj in obj_available[obj_available_mask == True]]

    data["task_list"] = {}
    data["task_list"]["train"] = training
    data["task_list"]["eval"] = evaluation
    data["total_step"] = 25000000
    data["h5_file_path"] = "./data/{scene}.h5"
    data["saving_period"] = 1000000
    data["max_t"] = 5
    data["action_size"] = 9

    train_param = {}
    train_param["cuda"] = True
    train_param["num_thread"] = 8
    train_param["gamma"] = 0.7
    train_param["seed"] = 1993
    train_param["reward"] = args["reward"]
    train_param["mask_size"] = 16

    data["train_param"] = train_param
    data["eval_param"] = {}
    data["eval_param"]["num_episode"] = 20
    data["method"] = args["method"]

    with open('param.json', 'w') as outfile:
        outfile.write(json.dumps(data, indent=4))

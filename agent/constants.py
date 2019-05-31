TASK_LIST = {
  'FloorPlan1': [{'position': {'x': -0.5, 'y': 0.9009992, 'z': 1.25}, 'rotation': {'x': 0.0, 'y': 270.0, 'z': 0.0}},
                 {'position': {'x': -0.25, 'y': 0.9009992, 'z': -1.25}, 'rotation': {'x': 0.0, 'y': 180.0, 'z': 0.0}}],# '43', '53', '69'],
  'FloorPlan2': [{'position': {'x': 1.25, 'y': 0.9009992, 'z': 0.0}, 'rotation': {'x': 0.0, 'y': 180.0, 'z': 0.0}}, 
                 {'position': {'x': -0.75, 'y': 0.9009992, 'z': 0.0}, 'rotation': {'x': 0.0, 'y': 270.0, 'z': 0.0}}],# '320', '384', '387'],
  # 'kitchen_02': ['90', '136', '157', '207', '329'],
  # 'living_room_08': ['92', '135', '193', '228', '254']
}

# Learning step before backpropagation
MAX_STEP = 5

#Approximate frame per agent
FRAME_PER_AGENT = 300000

#Total frame viewed
TOTAL_PROCESSED_FRAMES = FRAME_PER_AGENT * MAX_STEP

# Early stop can be triggered here
EARLY_STOP = 1000

# Use resnet to compute feature from observation (frame)
USE_RESNET = True

# EARLY_STOP = TOTAL_PROCESSED_FRAMES
# TOTAL_PROCESSED_FRAMES = 10 * 10**6 # 10 million frames


ACTION_SPACE_SIZE = 4
NUM_EVAL_EPISODES = 100
VERBOSE = True 
SAVING_PERIOD = 10 ** 6 // 200
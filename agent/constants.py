TASK_LIST = {
  'bathroom_02': ['26', '37'],# '43', '53', '69'],
  'bedroom_04': ['134', '264'],# '320', '384', '387'],
  # 'kitchen_02': ['90', '136', '157', '207', '329'],
  # 'living_room_08': ['92', '135', '193', '228', '254']
}
MAX_STEP = 5
FRAME_PER_AGENT = 300000
TOTAL_PROCESSED_FRAMES = FRAME_PER_AGENT * MAX_STEP
# TOTAL_PROCESSED_FRAMES = 10 * 10**6 # 10 million frames

ACTION_SPACE_SIZE = 4
NUM_EVAL_EPISODES = 100
VERBOSE = True
SAVING_PERIOD = 10 ** 6 // 200
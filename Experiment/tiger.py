from psychopy import core, visual, event, gui, data
from psychopy.visual.shape import ShapeStim
import random
import numpy as np
import csv

# 设置实验信息
expInfo = {'participant': '', 'session': '001'}
dlg = gui.DlgFromDict(dictionary=expInfo, title="Multi-Armed Bandit Task")
if not dlg.OK:
    core.quit()

participant = expInfo['participant']

# 设置数据文件
filename = f"{expInfo['participant']}_{expInfo['session']}_data.csv"
data_file = open(filename, 'w', newline='')
#dataFile.write('trial,chosen_arm,reward,response_time\n')
csv_writer = csv.writer(data_file)
# csv_writer.writerow(['Round', 'Option1', 'Option2', 'Reward1', 'Reward2', 'ChosenOption', 'ChosenReward'])
csv_writer.writerow(['', 'participant', 'block_label', 'trial_block', 'f_cor', 'f_inc', 'cor_option', 'inc_option', 'times_seen', 'rt', 'accuracy'])

# 创建窗口和刺激
win = visual.Window([800, 600], allowGUI=True, monitor='testMonitor', units='pix', color='white' ) 
# units: deg, pix, cm, norm, height
instructions = visual.TextStim(win, text="下面会出现两个选项，\n使左右键进行选择，\n尽可能最大化你的奖励。",color='black')
instructions.draw()
win.flip()
event.waitKeys(keyList=['space'])

fixation = visual.TextStim(win, text='+', color='black')
print(fixation.pos)
reward_text = [visual.TextStim(win, text='', pos=(-200, -100), color='black', height=30),
               visual.TextStim(win, text='', pos=(200, -100), color='black', height=30)]
highlight = visual.Rect(win, width=150, height=150, lineColor='black', lineWidth=3)

options = ['A', 'B', 'C', 'D']  # 四种可能的几何图案
# shapes_list = [
#     visual.Circle(win, radius=50, edges=100, pos=(0, 0), fillColor='blue', lineColor='black'),
#     visual.Rect(win, width=100, height=100, pos=(0, 0), fillColor='green', lineColor='black'),
#     visual.Polygon(win, edges=3, radius=60, pos=(0, 0), fillColor='red', lineColor='black'),
#     visual.Polygon(win, edges=6, radius=60, pos=(0, 0), fillColor='yellow', lineColor='black')
# ]
# random.shuffle(shapes_list)
# print(shapes_list)
# shapes = {
#     'A': shapes_list[0],
#     'B': shapes_list[1],
#     'C': shapes_list[2],
#     'D': shapes_list[3]
# }
# 替换几何图案为 PNG 图像
shapes_list = [
    visual.ImageStim(win, image=r'./new1.jpg', pos=(0, 0), size=(150, 150)),  # 替换为图片路径
    visual.ImageStim(win, image=r'./new2.jpg', pos=(0, 0), size=(150, 150)),
    visual.ImageStim(win, image=r'./new3.jpg', pos=(0, 0), size=(150, 150)),
    visual.ImageStim(win, image=r'./new4.jpg', pos=(0, 0), size=(150, 150)),
]

# 随机打乱图像列表以分配给选项
random.shuffle(shapes_list)

# 映射选项到图像
shapes = {
    'A': shapes_list[0],
    'B': shapes_list[1],
    'C': shapes_list[2],
    'D': shapes_list[3]
}
reward_params = {
    'A': {'mean': 36, 'std': 5},
    'B': {'mean': 40, 'std': 5},
    'C': {'mean': 50, 'std': 5},
    'D': {'mean': 54, 'std': 5},
}

blocks = 1
trials_per_block = 90
response_keys = {'left': 'q', 'right': 'p'}

#def generate_rewards(options):
#    return [int(round(np.random.normal(loc=reward_params[opt]['mean'], scale=reward_params[opt]['std']))) for opt in options]

def generate_reward(option):
    return int(round(np.random.normal(loc=reward_params[option]['mean'], scale=reward_params[option]['std'])))

import numpy as np

def generate_array_with_constraints(elements, counts, max_consecutive):
    """
    生成一个数组，随机排列，同时限制连续相同元素不超过 max_consecutive。
    
    :param elements: 列表，包含元素种类，例如 [1, 2, 3]
    :param counts: 列表，对应每种元素的数量，例如 [15, 15, 15]
    :param max_consecutive: 限制连续相同元素的最大数量
    :return: 满足条件的数组
    """
    # 创建初始数组
    array = np.repeat(elements, counts)
    np.random.shuffle(array)  # 先随机打乱
    
    # 如果不满足条件，重新排列
    while not is_valid_array(array, max_consecutive):
        np.random.shuffle(array)
    
    return array

def is_valid_array(array, max_consecutive):
    """
    检查数组是否满足连续相同元素不超过 max_consecutive。
    
    :param array: 待检查的数组
    :param max_consecutive: 限制连续相同元素的最大数量
    :return: 是否满足条件
    """
    count = 1
    for i in range(1, len(array)):
        if array[i] == array[i - 1]:
            count += 1
            if count > max_consecutive:
                return False
        else:
            count = 1
    return True

# 实验循环
for block in range(1, blocks + 1):
    total_pair_list = [["A", "B"], ["A", "C"], ["B", "D"], ["C", "D"], ["A", "D"], ["B", "C"]]
    use_pair_list = []
    use_pair_num_list = generate_array_with_constraints(elements=[0, 1, 2, 3, 4, 5], counts=[15, 15, 15, 15, 15, 15], max_consecutive=3)
    use_pair_reward_list = []
    cor_option_list = []
    inc_option_list = []
    f_cor_list = []
    f_inc_list = []
    rt_list = []
    accuracy_list = []
    symbols_times_seen_list = [0, 0, 0, 0, 0]
    times_seen_list = []
    for i in range(trials_per_block):
        random_int = use_pair_num_list[i]
        if random_int == 0:
            cor_option_list.append(2)
            inc_option_list.append(1)
        elif random_int == 1:
            cor_option_list.append(3)
            inc_option_list.append(1)
        elif random_int == 2:
            cor_option_list.append(4)
            inc_option_list.append(2)
        elif random_int == 3:
            cor_option_list.append(4)
            inc_option_list.append(3)
        elif random_int == 4:
            cor_option_list.append(4)
            inc_option_list.append(1)
        elif random_int == 5:
            cor_option_list.append(3)
            inc_option_list.append(2)
    
        symbols_times_seen_list[cor_option_list[-1]] += 1
        symbols_times_seen_list[inc_option_list[-1]] += 1
        current_times_seen = (symbols_times_seen_list[cor_option_list[-1]] + symbols_times_seen_list[inc_option_list[-1]]) // 2
        times_seen_list.append(current_times_seen)
            
        f_inc = generate_reward(total_pair_list[random_int][0])
        f_cor = generate_reward(total_pair_list[random_int][1])
        f_cor_list.append(f_cor)
        f_inc_list.append(f_inc)
        current_reward_list = [f_inc, f_cor]
        left_or_right = random.randint(0,1)
        if left_or_right == 0:
            use_pair_list.append(total_pair_list[random_int])
            use_pair_reward_list.append(current_reward_list)
        else:
            use_pair_list.append(total_pair_list[random_int][::-1])
            use_pair_reward_list.append(current_reward_list[::-1])
    print(f"Block {block} start:")
    
    for i in range(trials_per_block):
        left_option = use_pair_list[i][0]
        right_option = use_pair_list[i][1]
        left_reward = use_pair_reward_list[i][0]
        right_reward = use_pair_reward_list[i][1]
        
        # 1. 注视点阶段
        fixation.draw()
        win.flip()
        core.wait(random.uniform(0.75, 1.25))
        
        # 2. 选项展示阶段
        left_shape = shapes[left_option]
        right_shape = shapes[right_option]
        
        # 设置位置
        left_shape.pos = (-200, 0)  # 左侧位置
        right_shape.pos = (200, 0)  # 右侧位置

        # 绘制选项
        left_shape.draw()
        right_shape.draw()
        win.flip()
        
        start_time = core.getTime()
        
        keys = event.waitKeys(keyList=['left', 'right', 'escape'], timeStamped=True)
        if 'escape' in keys:
            break  # 提前退出实验

        key_pressed, response_time = keys[0]
        reaction_time = response_time - start_time
        rt_list.append(reaction_time)
        
        # 确定被试选择的选项
        chosen_option = left_option if key_pressed == 'left' else right_option
        if chosen_option == left_option:
            unchosen_option = right_option
        else:
            unchosen_option = left_option
        chosen_reward = left_reward if key_pressed == 'left' else right_reward
    
        if chosen_option < unchosen_option:
            accuracy = 0
        else:
            accuracy = 1
        accuracy_list.append(accuracy)
    
        # 3. 奖励展示阶段
        reward_text[0].setText(f"Reward: {left_reward:.2f}")
        reward_text[1].setText(f"Reward: {right_reward:.2f}")
        highlight.pos = left_shape.pos if key_pressed == 'left' else right_shape.pos  # 高亮框位置

        # 绘制奖励和高亮
        left_shape.draw()
        right_shape.draw()
        reward_text[0].draw()
        reward_text[1].draw()
        highlight.draw()
        win.flip()
        core.wait(2)  # 奖励展示持续 2 秒

        # 保存该轮数据
#        csv_writer.writerow([
#            i, left_option, right_option,
#            left_reward, right_reward,
#            chosen_option, chosen_reward
#        ])
        csv_writer.writerow([
            i, participant, 1, i+1, f_cor_list[i], f_inc_list[i],
            cor_option_list[i], inc_option_list[i],
            times_seen_list[i], reaction_time, accuracy
        ])
        data_file.flush()
        
data_file.close()
win.close()
core.quit()

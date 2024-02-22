import subprocess
import pytesseract
from PIL import Image
import cv2
import numpy as np
import time
# from skimage import feature

def subprocess_run(cmd_list):
    result = subprocess.run(cmd_list)
    # import pdb; pdb.set_trace()
    # print(result.stdout)
    time.sleep(0.5)


# 获取图片的长宽
def get_image_size(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    return width, height

# 连接手机
def connect_device():
    subprocess_run(['adb', 'devices'])


# 点击坐标
def click_coordinate(x, y):
    subprocess_run(['adb', 'shell', 'input', 'tap', str(x), str(y)])

# 截屏
def take_screenshot(new_name=None):
    subprocess_run(['adb', 'shell', 'screencap', '/sdcard/screenshot.png'])
    if new_name:
        subprocess_run(['adb', 'pull', '/sdcard/screenshot.png', new_name])
    else:
        subprocess_run(['adb', 'pull', '/sdcard/screenshot.png'])

# 识别图片中的文字
def recognize_text(image_path, region):
    image = Image.open(image_path)
    cropped_image = image.crop(region)
    text = pytesseract.image_to_string(cropped_image)
    return text

# 示例操作
def perform_operations():
    # 连接手机
    connect_device()

    # 截屏
    take_screenshot()

    # 识别图片中的文字
    screenshot_path = 'screenshot.png'
    region = (100, 200, 300, 400)  # 指定区域的左上角和右下角坐标
    text = recognize_text(screenshot_path, region)
    print("识别结果：", text)

# 对比图片
def compare_images(image1_path, image2_path, region, threshold=0.1):
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    cropped_image1 = image1[region[1]:region[3], region[0]:region[2]]
    cropped_image2 = image2[region[1]:region[3], region[0]:region[2]]
    difference = cv2.absdiff(cropped_image1, cropped_image2)
    # import pdb; pdb.set_trace()
    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    non_zero_pixels = np.count_nonzero(thresholded)
    total_pixels = thresholded.shape[0] * thresholded.shape[1]
    change_ratio = non_zero_pixels / total_pixels
    print(change_ratio)
    return change_ratio > threshold

# 判断图片2中是否存在图片1，并返回下标
def find_image(small_img_path, big_image_path, threshold):
    big_image = cv2.imread(big_image_path)
    small_image = cv2.imread(small_img_path)

    # 灰度化
    gray_big = cv2.cvtColor(big_image, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)

    # 提取关键点和描述子
    keypoints_big, descriptors_big = feature.sift(gray_big)
    keypoints_small, descriptors_small = feature.sift(gray_small)

    # 模糊匹配
    fuzziness = 0.5 # 模糊度参数，可以根据实际情况调整
    matches = feature.match_descriptors(descriptors_big, descriptors_small, method='bf', fuzziness=fuzziness)

    import pdb; pdb.set_trace()
    # 暴力匹配
    matches = np.where(matches == True)
    for i in range(len(matches[0])):
        point_big = keypoints_big[matches[0][i]].pt # 大图的匹配点坐标
        point_small = keypoints_small[matches[1][i]].pt # 小图的匹配点坐标

def click_bar():
    connect_device()
    compare_region = [100, 970, 750, 1030]
    # menu_img = cv2.imread("menu.png")

    take_screenshot()
    # screenshot_img = cv2.imread("screenshot.png")
    ret = compare_images("menu.png", "screenshot.png", compare_region, 0.4)
    while ret is True:
        click_coordinate(450, 1000)
        take_screenshot()
        # screenshot_img = cv2.imread("screenshot.png")
        ret = compare_images("menu.png", "screenshot.png", compare_region, 0.4)

    # click_coordinate(450, 1000)
    click_coordinate(670, 990)
    # return 

    # bar_img = cv2.imread("bar.png")
    # height, width, _ = bar_img.shape

    # take_screenshot()
    # ret = find_image("bar.png", "screenshot.png", 0.7)
    # # import pdb; pdb.set_trace()
    # while ret is None:
    #     click_coordinate(450, 1000)
    #     take_screenshot()
    #     ret = find_image("bar.png", "screenshot.png", 0.7)
    #     # import pdb; pdb.set_trace()

    #     print(ret)

    # click_coordinate(ret[0]+width/2, ret[1]+height/2)
# zoom_out()

# pixel_2340x1080_coords = [[770, 380], [920, 390], [1100, 380], [1200, 510]]
# coords = pixel_2340x1080_coords

home_coords = [770, 380]
zhitang_coords = [1650, 666]
shimo_coords = [1460, 760]
doufang_coords = [1800, 750]

tree_coords = [920, 390]
cedao_corrds = [560, 750]
jishe_corrds = [740, 650]
yangjuan_corrds = [860, 770]

bamboo_coords = [1100, 380]
bianzhi_corrds = [1100, 840]
wanju_corrds = [870, 820]

boil_coords = [1200, 510]
[1300, 430]
[1120, 320]

factory_queue_coords = [[1210, 950], [1040, 950], [890, 950], [700, 950], [530, 950], [380, 950]]
factory_mark_coords = [1800, 950]
factory_close_coords = [2150, 60]

def process_factory0(x, y, name):
    print("process_" + name)
    click_coordinate(x, y)

    for coord in factory_queue_coords:
        click_coordinate(coord[0], coord[1])
    
    for coord in factory_queue_coords:
        click_coordinate(factory_mark_coords[0], factory_mark_coords[1])
    
    click_coordinate(factory_close_coords[0], factory_close_coords[1])

def process_home():
    print("process_home")
    click_bar()
    click_coordinate(home_coords[0], home_coords[1])
    process_factory0(zhitang_coords[0], zhitang_coords[1], name="zhitang")

    click_bar()
    click_coordinate(home_coords[0], home_coords[1])
    process_factory0(shimo_coords[0], shimo_coords[1], name="shimo")

    click_bar()
    click_coordinate(home_coords[0], home_coords[1])
    process_factory0(doufang_coords[0], doufang_coords[1], name="doufang")

def harvest_tree():
    # import pdb; pdb.set_trace()
    click_coordinate(1200, 620)
    take_screenshot()
    compare_region = [1630, 900, 2020, 990]
    is_diff = compare_images("is_tree_ready.png", "screenshot.png", compare_region, 0.1)
    if is_diff is False:
        click_coordinate(factory_mark_coords[0], factory_mark_coords[1])
        click_coordinate(factory_mark_coords[0], factory_mark_coords[1])

    click_coordinate(factory_close_coords[0], factory_close_coords[1])

def process_factory1(x, y, name):
    print("process_" + name)
    click_coordinate(x, y)

    # for coord in factory_queue_coords:
    click_coordinate(1950, 940)
    click_coordinate(1650, 940)
    
    click_coordinate(factory_close_coords[0], factory_close_coords[1])

def process_tree():
    print("process_tree")
    click_bar()
    click_coordinate(tree_coords[0], tree_coords[1])
    harvest_tree()

    click_bar()
    click_coordinate(tree_coords[0], tree_coords[1])
    process_factory0(cedao_corrds[0], cedao_corrds[1], name="cedao")

    click_bar()
    click_coordinate(tree_coords[0], tree_coords[1])
    process_factory1(jishe_corrds[0], jishe_corrds[1], name="jishe")

    click_bar()
    click_coordinate(tree_coords[0], tree_coords[1])
    process_factory1(yangjuan_corrds[0], yangjuan_corrds[1], name="yangjuan")

def process_bamboo():
    print("process_bamboo")
    click_bar()
    click_coordinate(bamboo_coords[0], bamboo_coords[1])
    harvest_tree()

    # click_bar()
    # click_coordinate(bamboo_coords[0], bamboo_coords[1])
    # process_factory0(bianzhi_corrds[0], bianzhi_corrds[1], name="bianzhi")

    # click_bar()
    # click_coordinate(bamboo_coords[0], bamboo_coords[1])
    # process_factory0(wanju_corrds[0], wanju_corrds[1], name="wanju")

def process_boil():
    print("process_boil")
    click_bar()
    click_coordinate(boil_coords[0], boil_coords[1])
    harvest_tree()

    # click_bar()
    # click_coordinate(boil_coords[0], boil_coords[1])
    # process_factory0(bianzhi_corrds[0], bianzhi_corrds[1], name="bianzhi")

    # click_bar()
    # click_coordinate(boil_coords[0], boil_coords[1])
    # process_factory0(wanju_corrds[0], wanju_corrds[1], name="wanju")

# def process_tree():
#     print("process_tree")
#     click_bar()
#     click_coordinate(tree_coords[0], tree_coords[1])
#     harvest_tree()

def main():
    # take_screenshot()
    process_home()
    process_tree()
    process_bamboo()
    process_boil()

if __name__ == '__main__':
    main()  # next section explains the use of sys.exit

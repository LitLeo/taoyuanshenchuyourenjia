import subprocess
import pytesseract
from PIL import Image
import cv2
import numpy as np
import time

def subprocess_run(cmd_list):
    result = subprocess.run(cmd_list)
    # import pdb; pdb.set_trace()
    # print(result.stdout)
    time.sleep(1)


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
    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    non_zero_pixels = np.count_nonzero(thresholded)
    total_pixels = thresholded.shape[0] * thresholded.shape[1]
    change_ratio = non_zero_pixels / total_pixels
    return change_ratio > threshold

# 判断图片2中是否存在图片1，并返回下标
def find_image(template_path, image_path, threshold):
    template = cv2.imread(template_path, 0)
    image = cv2.imread(image_path, 0)

    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    # import pdb; pdb.set_trace()
    locations = np.where(result >= threshold)
    locations = list(zip(*locations[::-1]))

    if locations:
        # 获得查找结果的中间的坐标
        x, y = locations[-1][0], locations[-1][1]
        # height, width = template.shape
        # x = x + width / 2
        # y = y + height / 2
        return (x, y)
    else:
        return None

def click_bar():
    connect_device()
    bar_img = cv2.imread("bar.png")
    height, width, _ = bar_img.shape

    take_screenshot()
    ret = find_image("bar.png", "screenshot.png", 0.8)
    while ret is None:
        click_coordinate(400, 1000)
        take_screenshot()
        ret = find_image("bar.png", "screenshot.png", 0.8)
        # import pdb; pdb.set_trace()

        print(ret)

    click_coordinate(ret[0]+width/2, ret[1]+height/2)

# 屏幕放大
def zoom_in():
    subprocess_run(['adb', 'shell', 'input', 'touchscreen', 'swipe', 'x1', 'y1', 'x2', 'y2', 'duration'])

# 屏幕缩小
def zoom_out():
    subprocess_run(['adb', 'shell', 'input', 'touchscreen', 'swipe', 'x2', 'y2', 'x1', 'y1', 'duration'])

zoom_out()

click_bar()
take_screenshot(new_name="menu.png")

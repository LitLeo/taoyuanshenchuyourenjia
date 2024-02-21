import subprocess
import pytesseract
from PIL import Image
import cv2
import numpy as np

# 获取图片的长宽
def get_image_size(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    return width, height
  
# 连接手机
def connect_device():
    subprocess.run(['adb', 'devices'])

# 点击坐标
def click_coordinate(x, y):
    subprocess.run(['adb', 'shell', 'input', 'tap', str(x), str(y)])

# 截屏
def take_screenshot():
    subprocess.run(['adb', 'shell', 'screencap', '/sdcard/screenshot.png'])
    subprocess.run(['adb', 'pull', '/sdcard/screenshot.png'])

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

# 示例操作
def perform_operations():
    # 连接手机
    connect_device()

    # 截屏
    take_screenshot()

    # 对比图片
    previous_screenshot_path = 'previous_screenshot.png'
    current_screenshot_path = 'screenshot.png'
    region = (100, 200, 300, 400)  # 指定区域的左上角和右下角坐标
    change_threshold = 0.1  # 设置变化的阈值
    has_changed = compare_images(previous_screenshot_path, current_screenshot_path, region, change_threshold)
    if has_changed:
        print("区域发生较大变化")
    else:
        print("区域未发生较大变化")

    # 更新之前的截屏图片
    subprocess.run(['mv', current_screenshot_path, previous_screenshot_path])

# 屏幕放大
def zoom_in():
    subprocess.run(['adb', 'shell', 'input', 'touchscreen', 'swipe', 'x1', 'y1', 'x2', 'y2', 'duration'])

# 屏幕缩小
def zoom_out():
    subprocess.run(['adb', 'shell', 'input', 'touchscreen', 'swipe', 'x2', 'y2', 'x1', 'y1', 'duration'])


# 在大图像中查找小图像
def find_image(template_path, image_path, threshold):
    template = cv2.imread(template_path, 0)
    image = cv2.imread(image_path, 0)

    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    locations = list(zip(*locations[::-1]))

    # 返回小图像在大图像中的下标列表
    return locations

# 示例操作
def perform_operations():
    template_path = 'small_image.png'  # 小图像的路径
    image_path = 'large_image.png'  # 大图像的路径
    threshold = 0.8  # 匹配的阈值

    # 在大图像中查找小图像
    results = find_image(template_path, image_path, threshold)
    if len(results) > 0:
        print("小图像在大图像中的下标：", results)
    else:
        print("未找到小图像")

# 示例操作
def perform_operations():
    # 连接手机
    connect_device()

    # 点击坐标 (x, y)
    x = 100
    y = 200
    click_coordinate(x, y)

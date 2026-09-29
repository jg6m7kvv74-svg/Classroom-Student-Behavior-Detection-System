# 配置文件 - 全局路径设置

import os

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据目录
DATA_DIR = os.path.join(ROOT_DIR, "Data")
TEST_DATA_DIR = os.path.join(DATA_DIR, "test")
VAL_DATA_DIR = os.path.join(DATA_DIR, "val")
TEST_IMAGES_DIR = os.path.join(TEST_DATA_DIR, "images")
TEST_LABELS_DIR = os.path.join(TEST_DATA_DIR, "labels")
VAL_IMAGES_DIR = os.path.join(VAL_DATA_DIR, "images")
VAL_LABELS_DIR = os.path.join(VAL_DATA_DIR, "labels")

# 模型权重目录
WEIGHTS_DIR = os.path.join(ROOT_DIR, "weights")
DETECT_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, "detect")

# 默认模型路径
DEFAULT_MODEL = "D:\\YOLO\\ultralytics\\runs\\detect\\BSAM\\add_att_BSAM_origin\\weights\\best.pt"

# 批量检测输出目录
BATCH_OUTPUT_DIR = os.path.join(ROOT_DIR, "Folder batch detection")

# 测试结果目录
TEST_RESULTS_DIR = os.path.join(ROOT_DIR, "test_results")

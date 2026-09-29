import sys, warnings, os, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', message=".*requires.*Extramodule.*")
warnings.filterwarnings('ignore', message=".*gbk.*")



import cv2
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from PySide6.QtWidgets import *
from PySide6.QtGui import QPixmap, QImage, QFont
from PySide6.QtCore import Qt, QTimer

import torch
# from ultralytics.nn.modules.conv import CBAM, ChannelAttention, SpatialAttention
from ultralytics.nn.Extramodule import *
import ultralytics.nn.tasks
import ultralytics.nn.modules
import ultralytics.nn.modules.block
sys.modules['__main__'].CBAM = CBAM
sys.modules['__main__'].ChannelAttention = ChannelAttention
sys.modules['__main__'].SpatialAttention = SpatialAttention
setattr(ultralytics.nn.modules, 'CBAM', CBAM)
setattr(ultralytics.nn.modules.block, 'CBAM', CBAM)
setattr(ultralytics.nn.tasks, 'CBAM', CBAM)

_original_load = torch.load
def safe_load(*args, **kwargs):
    if 'weights_only' not in kwargs: kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = safe_load

from ultralytics import YOLO

CLASS_CN_MAP = {"tingke":"听课", "kanshouji":"看手机", "ditou":"低头", "zhanli":"站立", "shuijiao":"睡觉", "unknown":"未知类别"}

from config import DEFAULT_MODEL, BATCH_OUTPUT_DIR
from coze_api_module_v4 import call_coze_api_with_spatial, analyze_spatial_distribution
MODEL_CACHE = None

def markdown_to_html(text):
    """
    将Markdown文本转换为HTML
    
    Args:
        text: Markdown文本
        
    Returns:
        HTML文本
    """
    # 简单的Markdown替换，不依赖外部库
    # 处理标题
    text = text.replace('# ', '<h1>').replace('\n', '</h1>\n')
    text = text.replace('## ', '<h2>').replace('\n', '</h2>\n')
    text = text.replace('### ', '<h3>').replace('\n', '</h3>\n')
    # 处理粗体和斜体
    text = text.replace('**', '<b>').replace('**', '</b>')
    text = text.replace('*', '<i>').replace('*', '</i>')
    # 处理换行
    text = text.replace('\n', '<br>')
    # 处理列表
    text = text.replace('- ', '<li>').replace('\n', '</li>\n')
    # 处理代码块
    text = text.replace('```', '<pre><code>').replace('```', '</code></pre>')
    return text

LIGHT_STYLE = """
QMainWindow, QWidget { background-color: #f8f5f0; color: #333333; font-family: "微软雅黑"; font-size: 10pt; }
QGroupBox { border: 1px solid #d3b88c; border-radius: 6px; margin-top: 10px; padding-top: 15px; font-weight: bold; color: #8b5a2b; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #8b5a2b; }
QLineEdit, QComboBox { background-color: #ffffff; border: 1px solid #d3b88c; border-radius: 4px; padding: 6px; color: #333333; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox::down-arrow { image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJNMSAxTDYgNUwxMSAxIiBzdHJva2U9IiM4YjVhMmIiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0ibm9uZSIvPjwvc3ZnPg==); }
QComboBox QAbstractItemView { background-color: #ffffff; color: #333333; border: 1px solid #d3b88c; selection-background-color: #e8d7c3; }
QPushButton { background-color: #8b5a2b; border: 1px solid #6b4423; border-radius: 4px; padding: 10px; color: #ffffff; font-size: 11pt; }
QPushButton:hover { background-color: #a67c52; }
QPushButton:pressed { background-color: #6b4423; }
QLabel { color: #333333; }
QScrollArea { border: 1px solid #d3b88c; border-radius: 4px; background-color: #f8f5f0; }
QScrollArea QWidget { background-color: #f8f5f0; }
QScrollBar:vertical { background-color: #f8f5f0; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background-color: #d3b88c; border-radius: 5px; min-height: 20px; }
QScrollBar:horizontal { background-color: #f8f5f0; height: 10px; margin: 0; }
QScrollBar::handle:horizontal { background-color: #d3b88c; border-radius: 5px; min-width: 20px; }
QTextEdit { background-color: #ffffff; border: 1px solid #d3b88c; border-radius: 4px; padding: 10px; color: #333333; font-size: 10pt; line-height: 1.6; }
QSplitter::handle { background-color: #d3b88c; width: 2px; }
"""

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("课堂行为检测系统")
        self.setGeometry(100,100,1600,1000)
        self.setMinimumSize(1400,900)
        self.orig_img_cache = self.ann_img_cache = None
        self.batch_image_files, self.batch_current_index, self.batch_folder_path, self.batch_results = [], 0, "", {}
        self.camera_running, self.cap, self.timer = False, None, None
        self.load_default_model()
        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0,0,0,0)

        title_bar = QWidget()
        title_bar.setFixedHeight(80)
        title_layout = QVBoxLayout(title_bar)
        title_layout.setContentsMargins(0,0,0,0)
        bg_label = QLabel()
        bg_label.setFixedHeight(80)
        bg_label.setStyleSheet("background-color: #8b5a2b;")
        title_layout.addWidget(bg_label)
        main_layout.addWidget(title_bar)

        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setSpacing(15)
        content_layout.setContentsMargins(15,0,15,15)
        main_layout.addWidget(content_widget, stretch=1)

        left_widget = QWidget()
        left_widget.setMaximumWidth(320)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(5,0,5,5)
        content_layout.addWidget(left_widget)

        model_group = QGroupBox("模型配置")
        model_layout = QVBoxLayout(model_group)
        model_layout.setSpacing(8)
        model_layout.addWidget(QLabel("模型路径:"))
        self.model_edit = QLineEdit(DEFAULT_MODEL)
        model_layout.addWidget(self.model_edit)
        btn_model = QPushButton("选择模型文件")
        btn_model.clicked.connect(self.select_model)
        model_layout.addWidget(btn_model)
        left_layout.addWidget(model_group)

        param_group = QGroupBox("检测参数")
        param_layout = QVBoxLayout(param_group)
        param_layout.setSpacing(8)
        param_layout.addWidget(QLabel("置信度阈值:"))
        self.conf_combo = QComboBox()
        self.conf_combo.addItems(["0.2","0.3","0.4","0.5","0.6","0.7","0.8"])
        self.conf_combo.setCurrentText("0.5")
        param_layout.addWidget(self.conf_combo)
        left_layout.addWidget(param_group)

        opt_group = QGroupBox("检测操作")
        opt_layout = QVBoxLayout(opt_group)
        opt_layout.setSpacing(8)
        opt_layout.addWidget(QLabel("检测模式:"))
        self.detect_mode_combo = QComboBox()
        self.detect_mode_combo.addItems(["单张图片","文件夹批量检测","视频检测"])
        self.detect_mode_combo.currentTextChanged.connect(self.on_detect_mode_changed)
        opt_layout.addWidget(self.detect_mode_combo)
        opt_layout.addWidget(QLabel("路径:"))
        self.img_edit = QLineEdit()
        opt_layout.addWidget(self.img_edit)
        self.btn_select = QPushButton("选择检测图片")
        self.btn_select.clicked.connect(self.select_img)
        opt_layout.addWidget(self.btn_select)
        btn_detect = QPushButton("开始检测")
        btn_detect.setFixedHeight(40)
        btn_detect.setStyleSheet("font-size:14px; font-weight:bold; background-color: #8b5a2b; color: #ffffff;")
        btn_detect.clicked.connect(self.start_detect)
        opt_layout.addWidget(btn_detect)
        self.btn_camera = QPushButton("开启摄像头")
        self.btn_camera.setFixedHeight(35)
        self.btn_camera.clicked.connect(self.toggle_camera)
        opt_layout.addWidget(self.btn_camera)
        self.batch_nav_widget = QWidget()
        batch_nav_layout = QHBoxLayout(self.batch_nav_widget)
        batch_nav_layout.setContentsMargins(0,0,0,0)
        self.btn_prev = QPushButton("上一张")
        self.btn_prev.clicked.connect(self.show_prev_image)
        self.btn_prev.setEnabled(False)
        batch_nav_layout.addWidget(self.btn_prev)
        self.lbl_batch_info = QLabel("0 / 0")
        self.lbl_batch_info.setAlignment(Qt.AlignCenter)
        batch_nav_layout.addWidget(self.lbl_batch_info)
        self.btn_next = QPushButton("下一张")
        self.btn_next.clicked.connect(self.show_next_image)
        self.btn_next.setEnabled(False)
        batch_nav_layout.addWidget(self.btn_next)
        self.batch_nav_widget.setVisible(False)
        opt_layout.addWidget(self.batch_nav_widget)
        left_layout.addWidget(opt_group)
        left_layout.addStretch()

        center_widget = QWidget()
        center_layout = QHBoxLayout(center_widget)
        center_layout.setSpacing(10)
        center_layout.setContentsMargins(0,0,0,0)
        content_layout.addWidget(center_widget, stretch=3)

        left_img_widget = QWidget()
        left_img_layout = QVBoxLayout(left_img_widget)
        left_img_layout.setSpacing(5)
        left_img_layout.setContentsMargins(0,0,0,0)
        self.img_splitter = QSplitter(Qt.Horizontal)
        self.img_splitter.setHandleWidth(2)

        orig_panel = QWidget()
        orig_layout = QVBoxLayout(orig_panel)
        orig_layout.setContentsMargins(0,0,0,0)
        orig_layout.setSpacing(2)
        self.label_orig_title = QLabel("原图")
        self.label_orig_title.setAlignment(Qt.AlignCenter)
        self.label_orig_title.setFont(QFont("微软雅黑",10,QFont.Bold))
        self.label_orig_title.setStyleSheet("color: #8b5a2b;")
        self.scroll_orig = QScrollArea()
        self.scroll_orig.setWidgetResizable(True)
        self.scroll_orig.setStyleSheet("QScrollArea { border:1px solid #d3b88c; border-radius:4px; }")
        self.label_orig = QLabel("请选择图片")
        self.label_orig.setAlignment(Qt.AlignCenter)
        self.label_orig.setStyleSheet("color: #333333; font-size:14px; background-color: rgba(255,255,255,180); padding:10px;")
        self.scroll_orig.setWidget(self.label_orig)
        orig_layout.addWidget(self.label_orig_title)
        orig_layout.addWidget(self.scroll_orig)
        self.label_orig_filename = QLabel("")
        self.label_orig_filename.setAlignment(Qt.AlignCenter)
        self.label_orig_filename.setStyleSheet("font-size:10pt; color: #666;")
        orig_layout.addWidget(self.label_orig_filename)
        self.img_splitter.addWidget(orig_panel)

        ann_panel = QWidget()
        ann_layout = QVBoxLayout(ann_panel)
        ann_layout.setContentsMargins(0,0,0,0)
        ann_layout.setSpacing(2)
        self.label_ann_title = QLabel("检测结果图")
        self.label_ann_title.setAlignment(Qt.AlignCenter)
        self.label_ann_title.setFont(QFont("微软雅黑",10,QFont.Bold))
        self.label_ann_title.setStyleSheet("color: #8b5a2b;")
        self.scroll_ann = QScrollArea()
        self.scroll_ann.setWidgetResizable(True)
        self.scroll_ann.setStyleSheet("QScrollArea { border:1px solid #d3b88c; border-radius:4px; }")
        self.label_ann = QLabel("等待检测")
        self.label_ann.setAlignment(Qt.AlignCenter)
        self.label_ann.setStyleSheet("color: #333333; font-size:14px; background-color: rgba(255,255,255,180); padding:10px;")
        self.scroll_ann.setWidget(self.label_ann)
        ann_layout.addWidget(self.label_ann_title)
        ann_layout.addWidget(self.scroll_ann)
        self.label_ann_filename = QLabel("")
        self.label_ann_filename.setAlignment(Qt.AlignCenter)
        self.label_ann_filename.setStyleSheet("font-size:10pt; color: #666;")
        ann_layout.addWidget(self.label_ann_filename)
        self.img_splitter.addWidget(ann_panel)
        self.img_splitter.setCollapsible(0,False)
        self.img_splitter.setCollapsible(1,False)
        self.img_splitter.setStretchFactor(0,1)
        self.img_splitter.setStretchFactor(1,1)
        left_img_layout.addWidget(self.img_splitter, stretch=1)

        analysis_widget = QWidget()
        analysis_layout = QVBoxLayout(analysis_widget)
        analysis_layout.setContentsMargins(0,0,0,0)
        analysis_layout.setSpacing(2)
        analysis_title = QLabel("课堂建议")
        analysis_title.setStyleSheet("font-size:12px; font-weight:bold; color: #8b5a2b;")
        analysis_layout.addWidget(analysis_title)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("等待AI生成课堂建议...")
        self.result_text.setMinimumHeight(100)
        self.result_text.setStyleSheet("QTextEdit { border:1px solid #d3b88c; border-radius:4px; padding:8px; color: #333333; font-size:12px; }")
        analysis_layout.addWidget(self.result_text, stretch=1)
        left_img_layout.addWidget(analysis_widget, stretch=1)
        center_layout.addWidget(left_img_widget, stretch=2)

        # 创建包含饼图和条形图的容器
        chart_widget = QWidget()
        chart_widget.setStyleSheet("QWidget { background-color: #ffffff; border:1px solid #d3b88c; border-radius:4px; }")
        chart_layout = QVBoxLayout(chart_widget)
        chart_layout.setContentsMargins(10,10,10,5)
        chart_layout.setSpacing(10)
        
        # 饼图部分
        pie_title = QLabel("行为分布饼图")
        pie_title.setStyleSheet("font-size:14px; font-weight:bold; color: #8b5a2b; background-color: rgba(255,255,255,180); padding:5px; border-radius:3px;")
        chart_layout.addWidget(pie_title)
        self.pie_figure = plt.Figure(figsize=(4,3.5), dpi=90)
        self.pie_canvas = FigureCanvas(self.pie_figure)
        self.pie_canvas.setMinimumHeight(280)
        chart_layout.addWidget(self.pie_canvas)
        
        # 条形图部分
        line_title = QLabel("行为类别分布")
        line_title.setStyleSheet("font-size:14px; font-weight:bold; color: #8b5a2b; background-color: rgba(255,255,255,180); padding:5px; border-radius:3px;")
        chart_layout.addWidget(line_title)
        self.line_figure = plt.Figure(figsize=(4,2.5), dpi=90)
        self.line_canvas = FigureCanvas(self.line_figure)
        self.line_canvas.setMinimumHeight(180)
        chart_layout.addWidget(self.line_canvas)
        
        # 初始化条形图
        self.init_bar_chart()
        
        center_layout.addWidget(chart_widget, stretch=1)

        self.loading_widget = QWidget()
        self.loading_widget.setStyleSheet("background-color: #f8f5f0;")
        loading_layout = QVBoxLayout(self.loading_widget)
        loading_layout.setAlignment(Qt.AlignCenter)
        self.ai_label = QLabel("AI")
        self.ai_label.setAlignment(Qt.AlignCenter)
        self.ai_label.setStyleSheet("QLabel { color: #8b5a2b; font-size:32px; font-weight:bold; background-color:transparent; border:3px solid #8b5a2b; border-radius:50px; min-width:100px; min-height:100px; max-width:100px; max-height:100px; }")
        loading_layout.addWidget(self.ai_label, alignment=Qt.AlignCenter)
        self.rotation_angle = 0
        self.rotation_timer = QTimer()
        self.rotation_timer.timeout.connect(self.rotate_ai_circle)
        self.rotation_timer.start(50)
        self.loading_text = QLabel("等待检测...")
        self.loading_text.setAlignment(Qt.AlignCenter)
        self.loading_text.setStyleSheet("color: #333333; font-size:14px; margin-top:20px;")
        loading_layout.addWidget(self.loading_text)
        chart_layout.insertWidget(1, self.loading_widget)
        self.pie_canvas.setVisible(False)
        self.line_canvas.setVisible(False)
        self.init_pie_chart()

        right_widget = QWidget()
        right_widget.setMaximumWidth(350)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0,0,0,0)
        right_layout.setSpacing(15)
        content_layout.addWidget(right_widget)

        result_title = QLabel("检测结果")
        result_title.setStyleSheet("font-size:18px; font-weight:bold; color: #8b5a2b;")
        right_layout.addWidget(result_title)

        status_widget = QWidget()
        status_layout = QGridLayout(status_widget)
        status_layout.setContentsMargins(0,0,0,0)
        status_layout.setSpacing(10)
        lbl_state_label = QLabel("当前状态:")
        lbl_state_label.setStyleSheet("color: #333333;")
        status_layout.addWidget(lbl_state_label,0,0)
        self.lbl_status = QLabel("等待检测中...")
        self.lbl_status.setStyleSheet("color: #333333;")
        status_layout.addWidget(self.lbl_status,0,1)
        lbl_time_label = QLabel("用时:")
        lbl_time_label.setStyleSheet("color: #333333;")
        status_layout.addWidget(lbl_time_label,1,0)
        self.lbl_time = QLabel("--")
        self.lbl_time.setStyleSheet("color: #333333;")
        status_layout.addWidget(self.lbl_time,1,1)
        lbl_count_label = QLabel("检测行为目标:")
        lbl_count_label.setStyleSheet("color: #333333;")
        status_layout.addWidget(lbl_count_label,2,0)
        self.lbl_count = QLabel("0个")
        self.lbl_count.setStyleSheet("color: #333333;")
        status_layout.addWidget(self.lbl_count,2,1)
        right_layout.addWidget(status_widget)

        behavior_title = QLabel("行为类别统计")
        behavior_title.setStyleSheet("font-size:14px; font-weight:bold; color: #8b5a2b; margin-top:10px;")
        right_layout.addWidget(behavior_title)
        self.behavior_bars = {}
        behaviors = ["听课","看手机","低头","站立","睡觉"]
        for behavior in behaviors:
            bar_widget = QWidget()
            bar_layout = QHBoxLayout(bar_widget)
            bar_layout.setContentsMargins(0,0,0,0)
            bar_layout.setSpacing(10)
            lbl_name = QLabel(behavior)
            lbl_name.setFixedWidth(80)
            lbl_name.setStyleSheet("color: #333333;")
            bar_layout.addWidget(lbl_name)
            progress = QProgressBar()
            progress.setFixedHeight(20)
            progress.setStyleSheet("QProgressBar { border:1px solid #d3b88c; border-radius:3px; text-align:center; color: #333333; } QProgressBar::chunk { background-color: #8b5a2b; border-radius:3px; }")
            progress.setValue(0)
            bar_layout.addWidget(progress, stretch=1)
            lbl_value = QLabel("0%")
            lbl_value.setFixedWidth(50)
            lbl_value.setStyleSheet("color: #333333;")
            bar_layout.addWidget(lbl_value)
            right_layout.addWidget(bar_widget)
            self.behavior_bars[behavior] = (progress, lbl_value)
        right_layout.addStretch()

        action_layout = QHBoxLayout()
        self.btn_save = QPushButton("保存结果")
        self.btn_save.setFixedHeight(40)
        self.btn_save.clicked.connect(self.save_result)
        action_layout.addWidget(self.btn_save)
        btn_exit = QPushButton("退出")
        btn_exit.setFixedHeight(40)
        btn_exit.clicked.connect(self.close)
        action_layout.addWidget(btn_exit)
        right_layout.addLayout(action_layout)

        bottom_panel = QWidget()
        bottom_panel.setFixedHeight(20)
        bottom_panel.setStyleSheet("background-color: #f0f0f0; border-top:1px solid #ddd;")
        bottom_layout = QHBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(15,0,15,0)
        bottom_label = QLabel("课堂行为检测系统 v1.0")
        bottom_label.setStyleSheet("color: #666; font-size:11px;")
        bottom_layout.addWidget(bottom_label)
        bottom_layout.addStretch()
        main_layout.addWidget(bottom_panel)

    def load_default_model(self):
        global MODEL_CACHE
        if os.path.exists(DEFAULT_MODEL):
            try:
                MODEL_CACHE = YOLO(DEFAULT_MODEL)
                print(f"✅ 已自动加载默认模型: {DEFAULT_MODEL}")
            except Exception as e:
                print(f"❌ 加载默认模型失败: {str(e)}")
        else:
            print(f"⚠️ 默认模型文件不存在: {DEFAULT_MODEL}")

    def select_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择YOLO模型", "", "PyTorch模型 (*.pt)")
        if path:
            self.model_edit.setText(path)
            global MODEL_CACHE
            MODEL_CACHE = None

    def on_detect_mode_changed(self, mode):
        self.img_edit.clear()
        self.orig_img_cache = self.ann_img_cache = None
        self.label_orig.setText("请选择图片/文件夹/视频")
        self.label_ann.setText("等待检测")
        self.label_orig_filename.setText("")
        self.label_ann_filename.setText("")
        self.result_text.clear()
        self.batch_image_files, self.batch_current_index, self.batch_folder_path, self.batch_results = [], 0, "", {}
        self.batch_nav_widget.setVisible(False)
        if mode == "单张图片":
            self.btn_select.setText("选择检测图片")
            self.btn_select.clicked.disconnect()
            self.btn_select.clicked.connect(self.select_img)
        elif mode == "文件夹批量检测":
            self.btn_select.setText("选择图片文件夹")
            self.btn_select.clicked.disconnect()
            self.btn_select.clicked.connect(self.select_folder)
        elif mode == "视频检测":
            self.btn_select.setText("选择视频文件")
            self.btn_select.clicked.disconnect()
            self.btn_select.clicked.connect(self.select_video)

    def select_img(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择检测图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp)")
        if path:
            self.img_edit.setText(path)
            self.orig_img_cache, self.ann_img_cache = path, None
            self.label_ann.setText("等待检测")
            self.label_orig_filename.setText(os.path.basename(path))
            self.label_ann_filename.setText("")
            self.result_text.clear()
            self.refresh_img_display()

    def select_folder(self):
        path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if path:
            self.img_edit.setText(path)
            self.orig_img_cache = self.ann_img_cache = None
            self.label_orig.setText(f"已选择文件夹: {os.path.basename(path)}")
            self.label_ann.setText("等待批量检测")
            self.result_text.clear()

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self.img_edit.setText(path)
            self.orig_img_cache = self.ann_img_cache = None
            self.label_orig.setText(f"已选择视频: {os.path.basename(path)}")
            self.label_ann.setText("等待视频检测")
            self.result_text.clear()

    def display_img(self, scroll_area, label, img_source):
        view_size = scroll_area.viewport().size()
        if view_size.width() <=0 or view_size.height() <=0: return
        if isinstance(img_source, str):
            pixmap = QPixmap(img_source)
        else:
            rgb_img = cv2.cvtColor(img_source, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            qimg = QImage(rgb_img.data, w, h, ch*w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
        scaled_pixmap = pixmap.scaled(view_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)
        label.setFixedSize(scaled_pixmap.size())
        scroll_area.setFixedHeight(scaled_pixmap.height()+4)

    def refresh_img_display(self):
        if self.orig_img_cache is not None: self.display_img(self.scroll_orig, self.label_orig, self.orig_img_cache)
        if self.ann_img_cache is not None: self.display_img(self.scroll_ann, self.label_ann, self.ann_img_cache)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        total_width = self.img_splitter.width()
        self.img_splitter.setSizes([total_width//2, total_width//2])
        self.refresh_img_display()

    def init_pie_chart(self):
        self.pie_figure.clear()
        self.pie_figure.set_facecolor('#f8f5f0')
        ax = self.pie_figure.add_subplot(111)
        ax.set_facecolor('#f8f5f0')
        ax.set_aspect('equal')
        labels, sizes, colors = ['听课','看手机','低头','站立','睡觉'], [20,20,20,20,20], ['#4CAF50','#FF9800','#2196F3','#9C27B0','#F44336']
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize':14, 'color':'#333333'})
        ax.set_title('课堂行为分布', fontsize=16, fontweight='bold', color='#8b5a2b')
        self.pie_canvas.draw()
    
    def init_bar_chart(self):
        self.line_figure.clear()
        self.line_figure.set_facecolor('#f8f5f0')
        ax = self.line_figure.add_subplot(111)
        ax.set_facecolor('#f8f5f0')
        labels = ['听课','看手机','低头','站立','睡觉']
        values = [20,20,20,20,20]
        colors = ['#4CAF50','#FF9800','#2196F3','#9C27B0','#F44336']
        ax.bar(labels, values, color=colors)
        ax.set_title('行为类别分布', fontsize=14, fontweight='bold', color='#8b5a2b')
        ax.set_ylabel('数量', fontsize=12, color='#333333')
        ax.tick_params(axis='x', rotation=0, labelsize=10, colors='#333333')
        ax.tick_params(axis='y', labelsize=10, colors='#333333')
        self.line_canvas.draw()

    def update_pie_chart(self, cls_count):
        # 更新饼图
        self.pie_figure.clear()
        self.pie_figure.set_facecolor('#f8f5f0')
        ax = self.pie_figure.add_subplot(111)
        ax.set_facecolor('#f8f5f0')
        ax.set_aspect('equal')
        all_labels, all_colors = ['听课','看手机','低头','站立','睡觉'], ['#4CAF50','#FF9800','#2196F3','#9C27B0','#F44336']
        labels, sizes, colors = [], [], []
        for i, label in enumerate(all_labels):
            cnt = cls_count.get(label, 0)
            if cnt>0: labels.append(label); sizes.append(cnt); colors.append(all_colors[i])
        total = sum(sizes)
        if total ==0: labels, sizes, colors = all_labels, [20]*5, all_colors
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize':14, 'color':'#333333'})
        ax.set_title('课堂行为分布', fontsize=16, fontweight='bold', color='#8b5a2b')
        self.pie_canvas.draw()
        
        # 更新条形图
        self.line_figure.clear()
        self.line_figure.set_facecolor('#f8f5f0')
        ax = self.line_figure.add_subplot(111)
        ax.set_facecolor('#f8f5f0')
        # 获取所有类别的数据
        line_labels = []
        line_values = []
        line_colors = []
        for i, label in enumerate(all_labels):
            cnt = cls_count.get(label, 0)
            line_labels.append(label)
            line_values.append(cnt)
            line_colors.append(all_colors[i])
        # 如果没有数据，使用默认值
        if all(v == 0 for v in line_values):
            line_values = [20, 20, 20, 20, 20]
        ax.bar(line_labels, line_values, color=line_colors)
        ax.set_title('行为类别分布', fontsize=14, fontweight='bold', color='#8b5a2b')
        ax.set_ylabel('数量', fontsize=12, color='#333333')
        ax.tick_params(axis='x', rotation=0, labelsize=10, colors='#333333')
        ax.tick_params(axis='y', labelsize=10, colors='#333333')
        self.line_canvas.draw()

    def rotate_ai_circle(self):
        self.rotation_angle = (self.rotation_angle +10) %360
        r = max(100, min(255, int(122+50*abs(math.sin(math.radians(self.rotation_angle))))))
        g = max(130, min(255, int(162+50*abs(math.sin(math.radians(self.rotation_angle+120))))))
        b = max(200, min(255, int(247+50*abs(math.sin(math.radians(self.rotation_angle+240))))))
        color = f"#{r:02x}{g:02x}{b:02x}"
        self.ai_label.setStyleSheet(f"QLabel {{ color: {color}; font-size:32px; font-weight:bold; background-color:transparent; border:3px solid {color}; border-radius:50px; min-width:100px; min-height:100px; max-width:100px; max-height:100px; }}")

    def toggle_camera(self):
        self.stop_camera() if self.camera_running else self.start_camera()

    def start_camera(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened(): self.lbl_status.setText("摄像头打开失败"); return
            self.camera_running = True
            self.lbl_status.setText("摄像头运行中...")
            self.btn_camera.setText("关闭摄像头")
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_camera_frame)
            self.timer.start(30)
        except Exception as e: self.lbl_status.setText(f"摄像头错误: {str(e)}")

    def stop_camera(self):
        self.camera_running = False
        if self.timer: self.timer.stop()
        if self.cap: self.cap.release(); self.cap = None
        self.lbl_status.setText("摄像头已关闭")
        self.btn_camera.setText("开启摄像头")
        self.label_orig.setText("请选择图片")
        self.label_ann.setText("等待检测")

    def update_camera_frame(self):
        if self.camera_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.orig_img_cache = frame
                self.display_img(self.scroll_orig, self.label_orig, frame)

    def save_result(self):
        if self.ann_img_cache is None: self.lbl_status.setText("没有可保存的检测结果"); return
        path, _ = QFileDialog.getSaveFileName(self, "保存检测结果", "result.jpg", "图片文件 (*.jpg *.png)")
        if path:
            try:
                cv2.imwrite(path, self.ann_img_cache)
                self.lbl_status.setText(f"结果已保存: {os.path.basename(path)}")
            except Exception as e: self.lbl_status.setText(f"保存失败: {str(e)}")

    def start_detect(self):
        model_path, input_path, detect_mode = self.model_edit.text().strip(), self.img_edit.text().strip(), self.detect_mode_combo.currentText()
        if not os.path.exists(model_path): self.result_text.setText("❌ 错误：模型文件不存在，请检查路径"); return
        if not os.path.exists(input_path): self.result_text.setText("❌ 错误：输入路径不存在，请检查路径"); return
        if detect_mode == "单张图片": self.detect_single_image(model_path, input_path)
        elif detect_mode == "文件夹批量检测": self.detect_folder(model_path, input_path)
        elif detect_mode == "视频检测": self.detect_video(model_path, input_path)

    def detect_single_image(self, model_path, img_path):
        self.result_text.clear(); self.result_text.append("🔄 正在执行检测，请稍候..."); QApplication.processEvents()
        start_time = time.time()
        try:
            global MODEL_CACHE
            if MODEL_CACHE is None: self.result_text.append(f"📦 正在加载模型: {os.path.basename(model_path)}"); QApplication.processEvents(); MODEL_CACHE = YOLO(model_path)
            model, img, conf = MODEL_CACHE, cv2.imread(img_path), float(self.conf_combo.currentText())
            results = model(img, conf=conf, verbose=False)
            ann_img = results[0].plot()
            self.ann_img_cache, self.orig_img_cache = ann_img, img_path
            cls_count = {}
            boxes = []
            if len(results[0].boxes)>0:
                for box in results[0].boxes:
                    cls_en = model.names[int(box.cls[0])]
                    cls_cn = CLASS_CN_MAP.get(cls_en, CLASS_CN_MAP["unknown"])
                    cls_count[cls_cn] = cls_count.get(cls_cn, 0)+1
                    # 收集检测框信息 [x1, y1, x2, y2, class_en]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    boxes.append([x1, y1, x2, y2, cls_en])
            self.refresh_img_display()
            self.label_orig_filename.setText(f"原图: {os.path.basename(img_path)}")
            self.label_ann_filename.setText(f"检测结果: {os.path.basename(img_path)}")
            self.lbl_status.setText("检测完成")
            self.lbl_count.setText(f"{len(results[0].boxes)}个")
            total_count = sum(cls_count.values()) if cls_count else 1
            for behavior, (progress, lbl_value) in self.behavior_bars.items():
                count = cls_count.get(behavior, 0)
                percentage = int((count/total_count)*100) if total_count>0 else 0
                progress.setValue(percentage); lbl_value.setText(f"{count}个")
            self.loading_widget.setVisible(False); self.pie_canvas.setVisible(True); self.line_canvas.setVisible(True); self.loading_text.setText("检测完成")
            self.update_pie_chart(cls_count)
            end_time = time.time(); self.lbl_time.setText(f"{end_time-start_time:.2f}秒")
            # 获取图像尺寸
            img_height, img_width = img.shape[:2]
            # 调用API生成课堂建议
            suggestion = self.generate_classroom_suggestion(cls_count, boxes, img_width, img_height)
            # 将Markdown转换为HTML并显示
            html = markdown_to_html(suggestion)
            self.result_text.setHtml(html)
        except Exception as e: self.result_text.setText(f"❌ 检测出错: {str(e)}")

    def detect_folder(self, model_path, folder_path):
        self.result_text.clear(); self.result_text.append("🔄 开始批量检测..."); QApplication.processEvents()
        start_time = time.time()
        image_extensions = ('.jpg','.jpeg','.png','.bmp')
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
        image_files.sort()
        if not image_files: self.result_text.setText("❌ 文件夹中没有找到图片文件"); return
        try:
            global MODEL_CACHE
            if MODEL_CACHE is None: self.result_text.append(f"📦 正在加载模型: {os.path.basename(model_path)}"); QApplication.processEvents(); MODEL_CACHE = YOLO(model_path)
            model, conf = MODEL_CACHE, float(self.conf_combo.currentText())
            total_images = len(image_files)
            output_folder = BATCH_OUTPUT_DIR; os.makedirs(output_folder, exist_ok=True)
            self.batch_image_files, self.batch_folder_path, self.batch_current_index, self.batch_results = image_files, folder_path, 0, {}
            all_results = []
            for idx, img_file in enumerate(image_files, 1):
                img_path = os.path.join(folder_path, img_file)
                self.result_text.append(f"\n[{idx}/{total_images}] 检测: {img_file}"); QApplication.processEvents()
                img = cv2.imread(img_path)
                results = model(img, conf=conf, verbose=False)
                ann_img = results[0].plot()
                output_path = os.path.join(output_folder, f"result_{img_file}")
                cv2.imwrite(output_path, ann_img)
                self.batch_results[img_file] = {'orig_path':img_path, 'ann_img':ann_img, 'results':results[0]}
                cls_count = {}
                for box in results[0].boxes:
                    cls_en = model.names[int(box.cls[0])]
                    cls_cn = CLASS_CN_MAP.get(cls_en, CLASS_CN_MAP["unknown"])
                    cls_count[cls_cn] = cls_count.get(cls_cn, 0)+1
                all_results.append({'file':img_file, 'count':len(results[0].boxes), 'classes':cls_count})
            total_cls_count = {}
            for result in all_results:
                for cls, cnt in result['classes'].items(): total_cls_count[cls] = total_cls_count.get(cls, 0)+cnt
            suggestion = self.generate_classroom_suggestion(total_cls_count)
            html = markdown_to_html(suggestion)
            self.result_text.setHtml(html)
            self.loading_widget.setVisible(False); self.pie_canvas.setVisible(True); self.line_canvas.setVisible(True); self.loading_text.setText("检测完成")
            self.update_pie_chart(total_cls_count)
            end_time = time.time(); self.lbl_time.setText(f"{end_time-start_time:.2f}秒")
            self.show_batch_image(0); self.batch_nav_widget.setVisible(True); self.update_batch_nav_buttons()
        except Exception as e: self.result_text.setText(f"❌ 批量检测出错: {str(e)}")

    def show_batch_image(self, index):
        if not self.batch_image_files or index<0 or index>=len(self.batch_image_files): return
        self.batch_current_index = index
        img_file = self.batch_image_files[index]
        result_data = self.batch_results[img_file]
        self.orig_img_cache, self.ann_img_cache = result_data['orig_path'], result_data['ann_img']
        self.label_orig_filename.setText(f"原图: {img_file}")
        self.label_ann_filename.setText(f"检测结果: {img_file}")
        self.lbl_batch_info.setText(f"{index+1} / {len(self.batch_image_files)}")
        self.refresh_img_display()
        self.update_batch_nav_buttons()
        self.show_batch_image_details(img_file)

    def update_batch_nav_buttons(self):
        total, current = len(self.batch_image_files), self.batch_current_index
        self.btn_prev.setEnabled(current>0)
        self.btn_next.setEnabled(current < total-1)

    def show_prev_image(self):
        if self.batch_current_index>0: self.show_batch_image(self.batch_current_index-1)

    def show_next_image(self):
        if self.batch_current_index < len(self.batch_image_files)-1: self.show_batch_image(self.batch_current_index+1)

    def show_batch_image_details(self, img_file):
        if img_file not in self.batch_results: return
        result_data = self.batch_results[img_file]
        results = result_data['results']
        cls_count = {}
        for box in results.boxes:
            cls_en = results.names[int(box.cls[0])]
            cls_cn = CLASS_CN_MAP.get(cls_en, CLASS_CN_MAP["unknown"])
            cls_count[cls_cn] = cls_count.get(cls_cn, 0)+1
        total_count = sum(cls_count.values())
        for behavior, (progress, lbl_value) in self.behavior_bars.items():
            count = cls_count.get(behavior, 0)
            percentage = (count/total_count*100) if total_count>0 else 0
            progress.setValue(int(percentage)); lbl_value.setText(f"{count}个")
        self.lbl_status.setText("检测完成")
        self.lbl_count.setText(f"{total_count}个")
        self.lbl_time.setText("--")
        suggestion = self.generate_classroom_suggestion(cls_count)
        html = markdown_to_html(suggestion)
        self.result_text.setHtml(html)
        self.loading_widget.setVisible(False); self.pie_canvas.setVisible(True); self.line_canvas.setVisible(True); self.loading_text.setText("检测完成")
        self.update_pie_chart(cls_count)

    def generate_classroom_suggestion(self, cls_count, boxes=None, img_width=1920, img_height=1080):
        """
        调用Coze API生成课堂建议
        
        Args:
            cls_count: 行为统计字典
            boxes: 检测框信息列表
            img_width: 图像宽度
            img_height: 图像高度
            
        Returns:
            课堂建议文本
        """
        try:
            # 构建行为数据
            behavior_data = cls_count
            
            # 构建空间分布数据
            if boxes:
                boxes_info = []
                for box in boxes:
                    x_center = (box[0] + box[2]) / 2  # 计算中心点x坐标
                    cls_cn = CLASS_CN_MAP.get(box[4], "未知")
                    boxes_info.append({"class": cls_cn, "x_center": x_center})
                spatial_data = analyze_spatial_distribution(boxes_info, img_width)
            else:
                # 如果没有检测框信息，使用默认空间分布
                spatial_data = {"左侧区域": {}, "中间区域": {}, "右侧区域": {}}
            
            # 调用Coze API
            suggestion = call_coze_api_with_spatial(behavior_data, spatial_data, img_width, img_height)
            return suggestion
        except Exception as e:
            return f"⚠️ API调用失败: {str(e)}"

    def detect_video(self, model_path, video_path):
        self.result_text.clear(); self.result_text.append("🔄 开始视频检测..."); QApplication.processEvents()
        start_time = time.time()
        try:
            global MODEL_CACHE
            if MODEL_CACHE is None: self.result_text.append(f"📦 正在加载模型: {os.path.basename(model_path)}"); QApplication.processEvents(); MODEL_CACHE = YOLO(model_path)
            model, conf = MODEL_CACHE, float(self.conf_combo.currentText())
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): self.result_text.setText("❌ 无法打开视频文件"); return
            fps, width, height, total_frames = int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.result_text.append(f"📹 视频信息: {width}x{height}, {fps}fps, {total_frames}帧"); QApplication.processEvents()
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(os.path.dirname(video_path), f"{video_name}_detected.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            frame_count, all_cls_count = 0, {}
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame_count +=1
                if frame_count%5 ==0 or frame_count ==1:
                    results = model(frame, conf=conf, verbose=False)
                    ann_frame = results[0].plot()
                    for box in results[0].boxes:
                        cls_en = model.names[int(box.cls[0])]
                        cls_cn = CLASS_CN_MAP.get(cls_en, CLASS_CN_MAP["unknown"])
                        all_cls_count[cls_cn] = all_cls_count.get(cls_cn, 0)+1
                    if frame_count%30 ==0:
                        progress = (frame_count/total_frames)*100
                        self.result_text.append(f"⏳ 处理进度: {frame_count}/{total_frames} ({progress:.1f}%)"); QApplication.processEvents()
                        self.ann_img_cache, self.orig_img_cache = ann_frame, frame
                        self.refresh_img_display()
                else:
                    results = model(frame, conf=conf, verbose=False)
                    ann_frame = results[0].plot()
                out.write(ann_frame)
            cap.release(); out.release()
            suggestion = self.generate_classroom_suggestion(all_cls_count)
            html = markdown_to_html(suggestion)
            self.result_text.setHtml(html)
            self.loading_widget.setVisible(False); self.pie_canvas.setVisible(True); self.line_canvas.setVisible(True); self.loading_text.setText("检测完成")
            self.update_pie_chart(all_cls_count)
            end_time = time.time(); self.lbl_time.setText(f"{end_time-start_time:.2f}秒")
        except Exception as e: self.result_text.setText(f"❌ 视频检测出错: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(LIGHT_STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
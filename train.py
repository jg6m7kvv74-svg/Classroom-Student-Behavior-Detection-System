import warnings
warnings.filterwarnings('ignore')

# from config import TRAIN_CONFIG, DATA_DIR, WEIGHTS_DIR
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('D:/YOLO/ultralytics/yaml_/ATT_yaml/ATT_v11/v11_BSAMAtt_backbone_neck.yaml')
    # model = model.load("yolo11n.pt")
    # model = YOLO("yolo11n.pt")
    model.train(data='traindata/dataset_pro_904x5_split/data.yaml',
                # cache=False,
                imgsz=640,
                epochs=100,
                batch=8,
                workers=4,
                device='0',
                optimizer='SGD', #设置优化器为SGD（随机梯度下降），用于模型参数更新
                name = "add_att_BSAM"
                # patience=50,     # 在训练时，如果经过50轮性能没有提升，则停止训练（早停机制）
                # close_mosaic=10,  # 设置在训练结束前多少轮关闭 Mosaic 数据增强，10 表示在训练的最后 10 轮中关闭 Mosaic
                )
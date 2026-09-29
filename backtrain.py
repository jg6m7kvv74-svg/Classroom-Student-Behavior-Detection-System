from ultralytics import YOLO
import torch.multiprocessing as mp

if __name__ == '__main__':
    # 【关键】Windows 多进程必须加这一行
    mp.freeze_support()

    # 1. 加载模型 (last.pt)
    model = YOLO(r"runs/pose/4_1_yolov8n_pose_from_pt_coco2/weights/last.pt")

    # 2. 继续训练
    results = model.train(
        resume=True,
        # 可选：如果之前没指定设备，可以在这里加上
        # device='0'
    )
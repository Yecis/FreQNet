from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("runs/DroneVehicle/train/FreQNet/weights/best.pt")
    
    model.val(data="dataset/DroneVehicle.yaml",batch=8)
    # 打印模型参数信息
    print(model.info())

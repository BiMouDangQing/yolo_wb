
from ultralytics import YOLO

# 加载预训练的 YOLO26n 模型
model = YOLO("yolo26n.pt")

# 对示例图像运行推理
results = model("https://ultralytics.com/images/bus.jpg")

# 显示带标注的结果
results[0].show()
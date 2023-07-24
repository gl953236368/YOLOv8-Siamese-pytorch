from ultralytics import YOLO
from PIL import Image

from utils.utils import get_root

"""
   抖音点选验证码 图片识别位置 
        数据：204 背景图
"""
class _yolov8_(object):
    """
    TODO: 需要更新新模型
    """
    _defaults = {
        "model_path": get_root() + "/model_data/yolov8_model/best.pt",
        "train_yaml": ""
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"


    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.load_model()

    def load_model(self):
        self.model = YOLO(self.model_path)

    def train_model(self):
        """
        训练 默认是cpu训练
        :return:
        """
        ## cmd
        # yolo task=detect mode=train model=yolov8n.pt data=douyin_captcha-1/data.yaml
        # epochs=100 imgsz=640 workers=4 batch=4

        # model = YOLO("yolov8n.yaml") # 新建 yolo 模型

        model = YOLO("yolov8n.pt") # 加载 预训练的 yolo 模型 没有会自动下载

        # model = YOLO('yolov8n.yaml')._load('yolov8n.pt')

        # 训练
        res = model.train(data='datasets/douyin_captcha-1/data.yaml', epochs=100, imgsz=640)

        # 验证
        metrics = model.val()

        # 转换格式
        success = model.export(fomat="onnx")


    def export(self):
        """
        导出模型类型
        :return:
        """
        model = YOLO('runs/detect/train6/weights/best.pt')

        model.export(format='onnx')


    def predict(self, path, like=0.8):
        """
        对目标图片验证 并输出结果
        :param like:
        :return:
        """

        source = Image.open(path)
        results = self.model.predict(source=source, save=False, imgsz=640, conf=0.25, show=False)
        coordinates = []
        for _box in results[0].boxes:
            box = _box.boxes # 识别的数据 tensor
            _class = self.model.names[int(_box.cls)] # 识别的类别
            box_list = box.numpy().tolist()[0]
            x_min, y_min, x_max, y_max, confidence, other = box_list
            if confidence > like: # 相似度在目标范围内
                coordinates.append(
                    {
                        "x_min": int(x_min),
                        "x_max": int(x_max),
                        "y_min": int(y_min),
                        "y_max": int(y_max),
                        "confidence": confidence,
                        "class": _class
                    }
                )
        return coordinates



if __name__ == '__main__':
    # export()
    c = _yolov8_().predict("../img/fg_1686044968659_pic.png")
    # pass

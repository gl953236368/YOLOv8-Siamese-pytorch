import cv2
import ddddocr
from PIL import Image
from siamese import Siamese
from utils.utils import cost_time, image2bytes, get_root
from yolov_pytorch.train import _yolov8_

class Recognize():
    """

    """
    _defaults = {
        # 加载需要的模型
        ## ---------- 孪生网络 模型
        "siamese_model": Siamese(),
        ## ---------- 物体分类 模型
        "yolov8_model": _yolov8_(),
        ## ---------- 文字orc 识别
        "dddocr_model": ddddocr.DdddOcr(old=True, show_ad=False),
        ## ---------- 图片是否保存
        "is_save": False,
        ## ---------- 图片保存位置 bg
        "tmp_dir_bg": get_root() + "/img",
        ## ---------- 图片保存位置 fg
        "tmp_dir_fg": get_root() + "/img/image_bg_tmp"
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)

        for name, value in kwargs.items():
            setattr(self, name, value)


    def recognition_location(self, path):
        """
        识别图片中文字的位置
        :param path:
        :return:
        """
        coordinates = self.yolov8_model.predict(path)
        # print(coordinates)
        return coordinates

    def recognition_char(self, path):
        """
        识别图片中的文字
        :param path:
        :param box:
        :return:
        """
        if self.is_save:
            with open(path, 'rb') as f:
                image = f.read()
        else:
            image = path

        res = self.dddocr_model.classification(image)
        # print("命中目标:" + res)
        return res

    def rectangle_image(self, bg_path):
        """
        识别文字区域并保存
        :param fg_path:
        :param bg_path:
        :return:
        """
        coordinates = self.recognition_location(bg_path)  # 识别 区域

        im = cv2.imread(bg_path)
        #
        for box in coordinates:
            x1, y1, x2, y2 = box["x_min"], box["y_min"], box["x_max"], box["y_max"]
            im = cv2.rectangle(im, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

        if self.is_save:
            cv2.imwrite(f"{self.tmp_dir_bg}/result.jpg", im)

        return coordinates

    def main(self, fg_path, bg_path):

        res = list(self.recognition_char(fg_path))  # 识别 目标文字

        coordinates = self.rectangle_image(bg_path)

        bg_image = Image.open(bg_path)
        need_click = []
        for k, box in enumerate(coordinates):
            x1, y1, x2, y2 = box["x_min"], box["y_min"], box["x_max"], box["y_max"]
            z = bg_image.crop((x1, y1, x2, y2))
            path = f"{self.tmp_dir_bg}/{k}.jpg"
            z.save(path)
            char = self.recognition_char(path) # 识别目标坐标图
            box["class"] = char
            if char in res:
                need_click.append(box)

        print(need_click)


    def _split_image_fg(self, bg_path, path_name, target):
        """
        滑块部分图片切割
        Args:
            bg_path:
            path_name:
            target:

        Returns:

        """
        bg_image = Image.open(bg_path)
        count = bg_image.size[0] / 130
        print(f"当前图片：{bg_path} 数量: {count}")
        bg_image_list = []
        for i in range(int(count)):
            x1, y1, x2, y2 = i * 130, 0, (i + 1) * 130, 120
            z = bg_image.crop((x1, y1, x2, y2))
            if self.is_save:
                bg_image_list.append(z)
                path = f"{target}/{path_name}_{i}_fg.jpg"
                z.save(path)
            else:
                bg_image_list.append(image2bytes(z))
        return bg_image_list



    def _split_image_bg(self, source, target, coordinates):
        """
        背景图坐标位置切割图片
        :param source:
        :param target:
        :param coordinates:
        :return:
        """
        bg_image = Image.open(source)
        bg_image_list = []
        for k, box in enumerate(coordinates):
            x1, y1, x2, y2 = box["x_min"], box["y_min"], box["x_max"], box["y_max"]
            z = bg_image.crop((x1, y1, x2, y2))
            if self.is_save:
                bg_image_list.append(z)
                path = f"{self.tmp_dir_bg}/{target}_{k}.jpg"
                z.save(path)
            else:
                bg_image_list.append(image2bytes(z))

        return bg_image_list


    def recognize_image(self, model, bg_image_list, fg_image_list, fg_name, coordinates):
        """

        Args:
            model:
            bg_image_list:
            fg_image_bg:
            fg_path:
            coordinates:

        Returns:

        """
        probability_dict = {}
        for i in range(len(fg_image_list)):
            if self.is_save:
                _fg_path = f"{self.tmp_dir_fg}/{fg_name}_{i}_fg.jpg"
                char = self.recognition_char(_fg_path)
            else:
                char = self.recognition_char(fg_image_list[i])
            probability_list = []
            for m in range(len(coordinates)):
                image_1 = fg_image_list[i]
                image_2 = bg_image_list[m]
                probability = model.detect_image(image_1, image_2)

                probability_list.append({
                    "index": m,
                    "probability":probability.numpy().tolist()[0],
                    "coordinate": coordinates[m]
                })

            probability_list.sort(key=lambda x:x["probability"])

            # print(probability_list)

            probability_dict[char] = {
                "result": probability_list,
                "max_probability": probability_list[-1]
            }

        return probability_dict

    @cost_time
    def similar_image(self, fg_path, bg_path):
        """

        Args:
            fg_path:
            bg_path:
            dirpath:

        Returns:

        """
        # a_fg_path = self.tmp_dir_bg + "/" + fg_path
        # a_bg_path = self.tmp_dir_bg + "/" + bg_path

        fg_name = fg_path.split("/")[-1]
        bg_name = bg_path.split("/")[-1]
        coordinates = self.rectangle_image(bg_path)
        fg_image_list = self._split_image_fg(fg_path, fg_name, self.tmp_dir_fg)
        bg_image_list = self._split_image_bg(bg_path, bg_name, coordinates)
        probability_dict = self.recognize_image(self.siamese_model, bg_image_list, fg_image_list, fg_name, coordinates)

        print(probability_dict)
        return probability_dict


if __name__ == '__main__':
    Recognize().similar_image("img/fg_1686044968659_pic.png", "img/1686044968659_pic.jpeg")
    # print()

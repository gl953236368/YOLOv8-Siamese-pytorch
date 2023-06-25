import time

import cv2
import ddddocr
from PIL import Image
from siamese import Siamese
from yolov_pytorch.train import predict


def recognition_location(path):
    """
    识别图片中文字的位置
    :param path:
    :return:
    """
    coordinates = predict(path)
    # print(coordinates)
    return coordinates

def recognition_char(path):
    """
    识别图片中的文字
    :param path:
    :param box:
    :return:
    """
    det = ddddocr.DdddOcr(old=True, show_ad=False)

    with open(path, 'rb') as f:
        image = f.read()
    res = det.classification(image)
    print("命中目标:" + res)
    return res

def rectangle_image(bg_path):
    """
    识别文字区域并保存
    :param fg_path:
    :param bg_path:
    :return:
    """
    coordinates = recognition_location(bg_path)  # 识别 区域

    im = cv2.imread(bg_path)
    #
    for box in coordinates:
        x1, y1, x2, y2 = box["x_min"], box["y_min"], box["x_max"], box["y_max"]
        im = cv2.rectangle(im, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
    cv2.imwrite("image_tmp/result.jpg", im)

    return coordinates

def main(fg_path, bg_path):

    res = list(recognition_char(fg_path))  # 识别 目标文字

    coordinates = rectangle_image(bg_path)

    bg_image = Image.open(bg_path)
    need_click = []
    for k, box in enumerate(coordinates):
        x1, y1, x2, y2 = box["x_min"], box["y_min"], box["x_max"], box["y_max"]
        z = bg_image.crop((x1, y1, x2, y2))
        path = f"image_tmp/{k}.jpg"
        z.save(path)
        char = recognition_char(path) # 识别目标坐标图
        box["class"] = char
        if char in res:
            need_click.append(box)

    print(need_click)





def _split_image_fg(bg_path, path_name, target):
    bg_image = Image.open(bg_path)
    count = bg_image.size[0] / 130
    print(f"当前图片：{bg_path} 数量: {count}")
    bg_image_list = []
    for i in range(int(count)):
        x1, y1, x2, y2 = i * 130, 0, (i + 1) * 130, 120
        z = bg_image.crop((x1, y1, x2, y2))
        # path = f"{target}/{path_name}_{i}_fg.jpg"
        bg_image_list.append(z)
        path = f"{target}/{i}_fg.jpg"
        z.save(path)
    return bg_image_list



def similar_image(fg_path, bg_path):
    start_time = time.time()
    coordinates = rectangle_image(bg_path)
    fg_image_bg = _split_image_fg(fg_path, "", "image_tmp")
    model = Siamese()

    bg_image = Image.open(bg_path)
    bg_image_list = []
    for k, box in enumerate(coordinates):
        x1, y1, x2, y2 = box["x_min"], box["y_min"], box["x_max"], box["y_max"]
        z = bg_image.crop((x1, y1, x2, y2))
        bg_image_list.append(z)
        path = f"image_tmp/{k}.jpg"
        z.save(path)

    probability_dict = {

    }
    for i in range(len(fg_image_bg)):
        fg_path = f"image_tmp/{i}_fg.jpg"
        char = recognition_char(fg_path)
        probability_list = []
        for m in range(len(coordinates)):
            bg_path = f"image_tmp/{m}.jpg"
            # probability = recognize(fg_path, bg_path)
            # image_1 = Image.open(fg_path)
            image_1 = fg_image_bg[i]
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

    print(probability_dict)
    end_time = time.time()
    print("耗时： ", end_time - start_time)


if __name__ == '__main__':
    similar_image("image_tmp/fg_1686044968659_pic.png", "image_tmp/1686044968659_pic.jpeg")



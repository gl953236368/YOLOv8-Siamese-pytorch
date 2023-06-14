from PIL import Image
import os
import shutil

from imageHelper import recognition_location

"""
    构造siamese所需要的样本
"""


def get_file(source):
    for bg_path in os.listdir(source):
        print(bg_path)

def copy_file(source, target):
    shutil.copy(source, target)

def craete_dir(source, target):
    for path_name in os.listdir(source):
        bg_path = source + "/" + path_name
        if "fg" not in path_name:
            continue

        text = recognition_char(bg_path)
        new_dir = target + f"/chapter_{text}"
        if os.path.exists(new_dir):
            print("文件存在 " + new_dir)
        else:
            os.mkdir(new_dir)

        copy_file(bg_path, new_dir+"/"+path_name)
        # break

def _split_image_fg(bg_path, path_name, target):
    bg_image = Image.open(bg_path)
    count = bg_image.size[0] / 130
    print(f"当前图片：{bg_path} 数量: {count}")
    for i in range(int(count)):
        x1, y1, x2, y2 = i * 130, 0, (i + 1) * 130, 120
        z = bg_image.crop((x1, y1, x2, y2))
        # path = f"{target}/{path_name}_{i}_fg.jpg"
        path = f"{target}/{i}_fg.jpg"
        z.save(path)
    return count

def split_image_fg(source, target):
    for path_name in os.listdir(source):
        bg_path = source + path_name
        try:
            _split_image_fg(bg_path, path_name, target)
        except Exception as e:
            print("识别异常 %s" % e)
            continue


def split_image(source, target):

    for path_name in os.listdir(source):

        bg_path = source + path_name

        # if(bg_path != "/Users/gl_px/Desktop/crawl/douyin_captcha_images/bg_pic/1685677383172_pic.jpeg"):
        #     continue

        try:
            coordinates = recognition_location(bg_path)  # 识别 区域
        except Exception as e:
            print("识别异常 %s" % e)
            continue

        print(f"当前图片：{bg_path} 数据长度：{len(coordinates)}")
        bg_image = Image.open(bg_path)

        for k, box in enumerate(coordinates):
            x1, y1, x2, y2 = box["x_min"], box["y_min"], box["x_max"], box["y_max"]
            z = bg_image.crop((x1, y1, x2, y2))
            path = f"{target}/{path_name}_{k}.jpg"
            z.save(path)

        # break


if __name__ == '__main__':
    target = ""
    source = ""
    # split_image(path, target)
    # split_image_fg(path, target)
    craete_dir(target, source)

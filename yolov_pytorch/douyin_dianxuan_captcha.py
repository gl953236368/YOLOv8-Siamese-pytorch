from roboflow import Roboflow
"""
    通过调用三方平台训练集合 api接口进行识别
    https://app.roboflow.com/captcha-jjugy/douyin_captcha
"""

## 下载在第三方标注的结果数据
rf = Roboflow(api_key="AwIc4PTl6I3cfXEsTAyF")
project = rf.workspace("captcha-jjugy").project("douyin_captcha")
dataset = project.version(1).download("yolov8")

## 直接调用第三方标注平台训练的模型
# rf = Roboflow(api_key="AwIc4PTl6I3cfXEsTAyF")
# project = rf.workspace().project("douyin_captcha")
# model = project.version(1).model

## 直接识别本地图片 返回数据
# print(model.predict("1685958100593_pic.jpeg", confidence=40, overlap=30).json())

## 直接本地识别图片 返回数据 并保存
# model.predict("1685958100593_pic.jpeg", confidence=40, overlap=30).save("prediction.jpg")

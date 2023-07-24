from bottle import run, app, request, route
import os
from imageHelper import Recognize
from utils.utils import cost_time


@route('/upload', method="GET")
def upload():
    return '''
        <form action="/upload" method="post" enctype="multipart/form-data">
            背景:<input type="file" name="upload_bg"></br>
            切片:<input type="file" name="upload_fg"></br>
            提交 <input type="submit" value="Upload"></br>
        </form>
    '''


@route("/upload", method="POST")
@cost_time
def get_position():
    upload_bg = request.files.get('upload_bg', '')
    upload_fg = request.files.get('upload_fg', '')
    if not upload_bg.filename.lower().endswith(('.png', '.jpg', '.jpeg')) or \
            not upload_fg.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return 'File extension not allowed!'
    save_path = get_save_path()
    upload_bg_path = '%s%s' % (save_path, upload_bg.filename)
    upload_fg_path = '%s%s' % (save_path, upload_fg.filename)

    if not os.path.exists(upload_bg_path):
        upload_bg.save(save_path)
        upload_fg.save(save_path)

    c = model.similar_image(upload_fg_path, upload_bg_path)

    return 'Upload OK. FilePath: %s' % (c)


def get_save_path():
    return "./static/"

def load_model():
    model = Recognize()
    return model




if __name__ == '__main__':
    model = load_model()
    run(host='0.0.0.0', port=8088)
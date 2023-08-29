import cv2
import os
import numpy as np
from PIL import Image


def frame2video(im_dir, video_dir, fps):
    im_list = os.listdir(im_dir)
    im_list.sort(key=lambda x: int(x.replace("img", "").split('.')[0]))
    img = Image.open(os.path.join(im_dir, im_list[0]))
    img_size = img.size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # opencv版本是3
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

    for i in im_list:
        im_name = os.path.join(im_dir + '\\' + i)
        frame = cv2.imdecode(np.fromfile(im_name, dtype=np.uint8), -1)
        videoWriter.write(frame)

    videoWriter.release()
    print('Done')


if __name__ == '__main__':
    
    cho = 2  # 1 read 2 val

    if cho == 1:
        im_dir = r'P:\Dataset_DETRAC\Insight-MVT_Annotation_Test\MVI_39371'
        dir_video = '/'
    elif cho == 2:
        im_dir = r'/tracker/demo_result/result_images/demo'
        dir_video = r'/tracker/demo_result/result_images'

    dir_list = os.listdir(im_dir)
    fps = 25
    video_dir = dir_video + '\\' + 'demo.mp4'  # 因为我要存的是mp4格式，所以改了点点代码
    frame2video(im_dir, video_dir, fps)

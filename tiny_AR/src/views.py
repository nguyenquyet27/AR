from django.http import HttpResponse, StreamingHttpResponse, HttpResponseServerError
from django.shortcuts import render

import cv2
import time

from .render import Render
from . import config

# Create your views here.


def stream():
    cam = cv2.VideoCapture(0)

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cam.set(cv2.CAP_PROP_FPS, 60)  # default was 30fps
    # print(cam.get(cv2.CAP_PROP_SHARPNESS))

    k = 0
    while True:
        start = time.time()
        _, frame_read = cam.read()

        # main dish
        render = Render(frame_read)
        img_after = render.set_preprocess()
        matches = render.feature_matching()

        if(len(matches) > config.MIN_MATCHES):
            k += 1
            render.estimate_homomatrix()
            continue
        else:
            print(
                'Not enough matches found - {}/{}'.format(len(matches), config.MIN_MATCHES))

        if(k >= 30):
            pass
        print('consume: ', time.time()-start)
        imgencode = cv2.imencode('.jpg', frame_read)[1]
        stringData = imgencode.tostring()
        yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
    del(cam)


def indexscreen(request, *args, **kwargs):
    return render(request, "streaming.html", {})


def VideoView(request, *args, **kwargs):
    print(args, kwargs)
    try:
        return StreamingHttpResponse(stream(), content_type="multipart/x-mixed-replace;boundary=frame")
    except HttpResponseServerError as e:
        return e

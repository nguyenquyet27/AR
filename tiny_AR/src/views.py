from django.http import HttpResponse, StreamingHttpResponse, HttpResponseServerError
from django.shortcuts import render

import cv2

# Create your views here.


def stream():
    cam = cv2.VideoCapture(0)

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cam.set(cv2.CAP_PROP_FPS, 60)  # default was 30fps
    # print(cam.get(cv2.CAP_PROP_SHARPNESS))

    while True:
        _, frame_read = cam.read()
        # print(frame_read.shape)
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
    except:
        return "error"

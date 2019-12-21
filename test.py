# Basic OBJ file viewer. needs objloader from:
#  Original por http://www.pygame.org/wiki/OBJFileLoader
#  Ajustes por altaruru: https://www.altaruru.com
#  >> correcciones a la carga desde directorios distintos a origen
#  >> reposicionamiento inicial objeto
# LMB + move: rotate
# RMB + move: pan
# Scroll wheel: zoom in/out
# import sys, pygame
# from pygame.locals import *
# from pygame.constants import *
# from OpenGL.GL import *
# from OpenGL.GLU import *

# # IMPORT OBJECT LOADER
# from objloader_simple import *

# pygame.init()
# viewport = (800,600)
# hx = viewport[0]/2
# hy = viewport[1]/2
# srf = pygame.display.set_mode(viewport, OPENGL | DOUBLEBUF)

# glLightfv(GL_LIGHT0, GL_POSITION,  (-40, 200, 100, 0.0))
# glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
# glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
# glEnable(GL_LIGHT0)
# glEnable(GL_LIGHTING)
# glEnable(GL_COLOR_MATERIAL)
# glEnable(GL_DEPTH_TEST)
# glShadeModel(GL_SMOOTH)           # most obj files expect to be smooth-shaded

# # LOAD OBJECT AFTER PYGAME INIT
# obj = OBJ("models/11571_Gingerbread_cookie_male_V2_l2.obj", swapyz=True)

# clock = pygame.time.Clock()

# glMatrixMode(GL_PROJECTION)
# glLoadIdentity()
# width, height = viewport
# gluPerspective(90.0, width/float(height), 1, 100.0)
# glEnable(GL_DEPTH_TEST)
# glMatrixMode(GL_MODELVIEW)

# #rx, ry = (0,0)
# rx, ry = (183,-361) 
# #tx, ty = (0,0)
# tx, ty = (0,-174)
# zpos = 15
# rotate = move = False
# while 1:
# 	clock.tick(30)
# 	for e in pygame.event.get():
# 		if e.type == QUIT:
# 			sys.exit()
# 		elif e.type == KEYDOWN and e.key == K_ESCAPE:
# 			sys.exit()
# 		elif e.type == MOUSEBUTTONDOWN:
# 			if e.button == 4: zpos = max(1, zpos-1)
# 			elif e.button == 5: zpos += 1
# 			elif e.button == 1: rotate = True
# 			elif e.button == 3: move = True
# 		elif e.type == MOUSEBUTTONUP:
# 			if e.button == 1: rotate = False
# 			elif e.button == 3: move = False
# 		elif e.type == MOUSEMOTION:
# 			i, j = e.rel
# 			if rotate:
# 				rx += i
# 				ry += j
# 				# print("rotate: x=%d y=%d z=%d; " % (rx,ry,zpos))
# 			if move:
# 				tx += i
# 				ty -= j
# 				# print("move: x=%d y=%d z=%d" % (tx,ty,zpos))

# 	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
# 	glLoadIdentity()

# 	# # RENDER OBJECT
# 	glTranslate(tx/20., ty/20., - zpos)
# 	glRotate(ry, 1, 0, 0)
# 	glRotate(rx, 0, 1, 0)
# 	glCallList(obj.gl_list)

# 	pygame.display.flip()


import cv2
import numpy as np
import process_func as pf
from objloader_simple import *
import math

from pygame.locals import *
from pygame.constants import *
from OpenGL.GL import *
from OpenGL.GLU import *

MIN_MATCHES = 200
def main():
    """
    This functions loads the target surface image,
    """
    homography = None 
    camera_parameters = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]])
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    model = cv2.imread('template/joker.jpg')
    obj = OBJ('models/fox.obj',swapyz=True)
    img1 = pf.image_proc(model, 1)
    kp_model, des_model = orb.detectAndCompute(img1, None)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        img2 = pf.image_proc(frame, 1)
        # print(frame.shape)
        if not ret:
            print ("Unable to capture video")
            return 
        kp_frame, des_frame = orb.detectAndCompute(img2, None)
        if (des_frame is None): 
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        matches = bf.match(des_model, des_frame)
        matches = sorted(matches, key=lambda x: x.distance)

        if (len(matches) > MIN_MATCHES):
            src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            homography = pf.computeHomography(src_pts ,dst_pts)
            h, w = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, homography)
            # frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            if homography is not None:
                try:
                    # obtain 3D projection matrix from homography matrix and camera parameters
                    projection = pf.projection_matrix(camera_parameters, homography)
                    # project cube or model
                    frame = pf.render(frame, obj, projection, img1, False)
                except:
                    pass
            
        else:
            print ("Not enough matches found - %d/%d" % (len(matches), MIN_MATCHES)) 
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0

if __name__ == '__main__':
    main()
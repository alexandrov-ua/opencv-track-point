import os
import sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def plotAndGetCoordinate(img):
    plt.rcParams["figure.autolayout"] = True
    point = [0,0]
    def mouse_event(event):
        nonlocal point
        point[0] = event.xdata
        point[1] = event.ydata
        plt.close()

    fig = plt.figure()
    _ = fig.canvas.mpl_connect('button_press_event', mouse_event)
    plt.imshow(img),plt.show()
    return point

def processImgs(img1, img2, point_to_track):
    MIN_MATCH_COUNT = 10
    FLANN_INDEX_KDTREE = 1

    img1gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(img1gray,None)
    kp2, des2 = sift.detectAndCompute(img2gray,None)
    
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    
    flann = cv.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1,des2,k=2)
    
    good = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
        M, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        pointTransform = cv.perspectiveTransform(point_to_track.reshape(-1,1,2),M)
        return pointTransform[0][0]
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        return point_to_track

file_patch = sys.argv[1]
if not os.path.isfile(file_patch):
    print(f"File {file_patch} is not exists.")
    exit(1)

cap = cv.VideoCapture(file_patch)
prev = None
point_to_track = np.float32([0, 0])
while True:
    ret, nxt = cap.read()
    if not ret:
        break

    if prev is None:
        point = plotAndGetCoordinate(nxt)
        point_to_track = np.float32(point)
    else:
        point_to_track = processImgs(prev, nxt, point_to_track)
        nxt = cv.circle(nxt, np.int32(point_to_track), 5, 255, -1)
        cv.imshow('frame', nxt)
    
    prev = nxt
    if cv.waitKey(1) & 0xFF == ord('q'):
        cv.destroyAllWindows()
        cap.release()




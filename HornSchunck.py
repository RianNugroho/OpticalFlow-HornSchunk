from scipy.ndimage.filters import convolve
import numpy as np
import cv2
import math


def HornSchunck(img1,img2, alpha, iteration):
    ##HSkernel ada di paper aslinya
    HornSchunckKernel=np.array([[1/12, 1/6, 1/12],
                   [            1/6,    0, 1/6],
                                [1/12, 1/6, 1/12]], dtype="float32")

    img1=np.array(img1,dtype="float32")
    img2 = np.array(img2, dtype="float32")

    sobelX=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype="float32")
    sobelY=np.array([[1,2,1],[0,0,0],[-1,-2,-1]],dtype="float32")

    Ix=convolve(img1,sobelX)
    Iy=convolve(img1,sobelY)
    It=np.subtract(img2,img1)

    U =np.zeros([img1.shape[0],img1.shape[1]])
    V =np.zeros([img1.shape[0], img1.shape[1]])

    for i in range(iteration):
        uAverage=convolve(U,HornSchunckKernel)
        vAverage=convolve(V,HornSchunckKernel)
        P=Ix*uAverage+Iy*vAverage+It
        D=alpha+Ix**2+Iy**2
        U=uAverage-Ix*P/D
        V=vAverage-Iy*P/D

    U=np.where(abs(U)<0.1,0,U)
    V = np.where(abs(V) < 0.1, 0, V)

    return U,V


def main():
    cap = cv2.VideoCapture('senam.mp4')
    ret, old_frame = cap.read()
    prev_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    while (1):
        ret, frame = cap.read()
        new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        u, v = HornSchunck(prev_gray, new_gray, 0.001, 10)



        for i in range(len(u)):
            for j in range(len(u[i])):
                if(math.sqrt(u[i][j]**2+v[i][j]**2)>=0.4):
                    frame[i][j]=[0,0,255]

        cv2.imshow('Magnitude', frame)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        prev_gray = new_gray

    cap.release()
    cv2.destroyAllWindows()

main()
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread, imsave
from scipy import signal
from time import time
from numba import jit

def jpeg(fil, qlist):
    # center pixel intensity
    fil -= 128

    N, M = fil.shape
    step = 8

    Q = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]])

    # lists to be returned
    fil_rek = []
    entropies = []

    for q in range(len(qlist)):
        # dct
        blokker_freq = np.zeros((N,M))
        for i in range(0, N, step):
            for j in range(0, M, step):
                blokk = np.zeros((step, step))
                blokk = fil[i:i+step, j:j+step]
                blokker_freq[i:i+step, j:j+step] = dct_jit(blokk, qlist, q, Q) 
        entropies.append(entropy(blokker_freq))

        # idct
        blokker_rom = np.zeros((N,M))
        for i in range(0, N, step):
            for j in range(0, M, step):
                blokk = np.zeros((step, step))
                blokk = blokker_freq[i:i+step, j:j+step]*(qlist[q]*Q)
                blokker_rom[i:i+step, j:j+step] = idct_jit(blokk) 
        fil_rek.append(blokker_rom+128)
    return fil_rek, entropies

@jit
# jit-able part of dct. Greatly decreases runtime
def dct_jit(blokk, qlist, q, Q):
    step = 8
    blokk_freq = np.zeros((step,step))
    for u in range(step):
        for v in range(step):
            summ = 0
            for x in range(step):
                for y in range(step):
                    summ += blokk[x, y] * np.cos(((2*x+1)*u*np.pi)/16)*np.cos(((2*y+1)*v*np.pi)/16)
            blokk_freq[u, v] = 0.25*c(u)*c(v)*summ
    return np.around(blokk_freq/(qlist[q]*Q))

@jit
# jit-able part of idct
def idct_jit(blokk):
    step = 8
    blokk_rom = np.zeros((step, step))
    for x in range(step):
        for y in range(step):
            summ = 0
            for u in range(step):
                for v in range(step):
                    summ += c(u)*c(v)*blokk[u, v]*np.cos(((2*x+1)*u*np.pi)/16)*np.cos(((2*y+1)*v*np.pi)/16)
            blokk_rom[x,y] = summ*0.25
    return blokk_rom

@jit
def c(a):
    return 1/np.sqrt(2) if a==0 else 1

@jit
def entropy(img):
    # using dict since the values don't fall between 0 and 255
    hist = {}
    N, M = img.shape
    unit = 1/(N*M)
    for i in range(N):
        for j in range(M):
            if img[i,j] in hist:
                hist[img[i,j]] += unit
            else:
                hist[img[i,j]] = unit
    summ = 0
    for p in hist:
        summ -= hist[p]*np.log2(hist[p])

    return round(summ, 4)

def main():
    fil = imread("uio.png", as_gray=True)
    qlist = [0.1, 0.5, 2, 8, 32]
    fil_rek, entropies = jpeg(fil, qlist)

    plt.subplot(321)
    plt.imshow(fil, cmap="gray")
    plt.title("Original image")

    for f in range(len(fil_rek)):
        imsave("jpeg"+str(f)+".jpg", fil_rek[f]) 
        plt.subplot(3,2,f+2)

        plt.imshow(fil_rek[f], cmap="gray")
        plt.title("Compression rate factor q = " + str(qlist[f]) + " entropy = " + str(entropies[f]))

    plt.show()

if __name__ == "__main__":
    main()

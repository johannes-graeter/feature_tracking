#!/usr/bin/env python2

import cv2
from matplotlib import pyplot as plt
import os

import matcher

def plot(features, img):
    
    for f in features:
        # cv2.circle(img,(int(f.u1p),int(f.v1p)), 1, [100],2,-1)
        cv2.circle(img,(int(f.u1c),int(f.v1c)), 2, [100],1,-1)
        cv2.line(img,(int(f.u1p),int(f.v1p)),(int(f.u1c),int(f.v1c)), [100], 1, 1)
    
    plt.imshow(img,cmap='gray')
    plt.show()

def main():
    params=matcher.MatcherParams()
    m=matcher.Matcher(params)
    
    assert(len(m.getMatches())==0)
    
    test_dir = os.path.dirname(os.path.realpath(__file__)) 
    I0=cv2.imread(os.path.join(test_dir, "000106.png",0))
    I1=cv2.imread(os.path.join(test_dir, "000107.png",0))
    
    matcher.pushBack(m,I0,False)
    matcher.pushBack(m,I1,False)
    
    m.matchFeatures(0)
    
    features=m.getMatches()

    print("number features="+str(len(features)))
    assert(len(features)==3306)

    plot(features,I1)

if __name__ == '__main__':
    main()


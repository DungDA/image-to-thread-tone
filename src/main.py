#!/usr/bin/env python

import os
import sys
import cv2
import numpy as np

# local import
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

# Parameters
imgRadius = 500     # Number of pixels that the image radius is resized to

initPin = 0         # Initial pin to start threading from
numPins = 512       # Number of pins on the circular loom
numLines = 2048     # Maximal number of lines

minLoop = 3         # Disallow loops of less than minLoop lines
lineWidth = 3       # The number of pixels that represents the width of a thread
lineWeight = 15     # The weight a single thread has in terms of "darkness"

dataroot = './examples'
savefolder = './results'

# Apply circular mask to image
def maskImage(image, radius):
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = x**2 + y**2 > radius**2
    image[mask] = 0

    return image

# Compute coordinates of loom pins
def pinCoords(radius, numPins=200, offset=0, x0=None, y0=None):
    alpha = np.linspace(0 + offset, 2*np.pi + offset, numPins + 1)

    if (x0 == None) or (y0 == None):
        x0 = radius + 1
        y0 = radius + 1

    coords = []
    for angle in alpha[0:-1]:
        x = int(x0 + radius*np.cos(angle))
        y = int(y0 + radius*np.sin(angle))

        coords.append((x, y))
    return coords

# Compute a line mask
def linePixels(pin0, pin1):
    length = int(np.hypot(pin1[0] - pin0[0], pin1[1] - pin0[1]))

    x = np.linspace(pin0[0], pin1[0], length)
    y = np.linspace(pin0[1], pin1[1], length)

    return (x.astype(np.int)-1, y.astype(np.int)-1)


def training_image_to_draw():
    exp = 'pretrained'
    imgsize = 512
    epoch = '200'
    gpu_id = '0'
    os.system(
        'python3 image-to-draw.py --dataroot %s --name %s --model test_3styles --output_nc 1 --no_dropout --num_test 1000 --epoch %s --imagefolder %s --crop_size %d --load_size %d --gpu_ids %s' % (
            dataroot, exp, epoch, 'images3styles', imgsize, imgsize, gpu_id
        )
    )


def convert_to_pins():
    # Auto find last result
    imgPath = os.path.join(savefolder, 'pretrained', 'test_200', 'images3styles', 'image_fake3.png')

    # Load image
    image = cv2.imread(imgPath)
    print("[+] loaded " + imgPath + " for threading..")

    # Crop image
    height, width = image.shape[0:2]
    minEdge= min(height, width)
    topEdge = int((height - minEdge)/2)
    leftEdge = int((width - minEdge)/2)
    imgCropped = image[topEdge:topEdge+minEdge, leftEdge:leftEdge+minEdge]
    cv2.imwrite(os.path.join(savefolder, 'cropped.png'), imgCropped)

    # Resize image
    imgSized = cv2.resize(imgCropped, (2*imgRadius + 1, 2*imgRadius + 1))

    # Mask image
    imgMasked = maskImage(imgSized, imgRadius)
    cv2.imwrite('./masked.png', imgMasked)

    print("[+] image preprocessed for threading..")

    # Define pin coordinates
    coords = pinCoords(imgRadius, numPins)
    height, width = imgMasked.shape[0:2]

    # image result is rendered to
    imgResult = 255 * np.ones((height, width))

    # Initialize variables
    i = 0
    lines = []
    previousPins = []
    oldPin = initPin
    lineMask = np.zeros((height, width))

    imgResult = 255 * np.ones((height, width))

    # Loop over lines until stopping criteria is reached
    for line in range(numLines):
        i += 1
        bestLine = 0
        oldCoord = coords[oldPin]

        # Loop over possible lines
        for index in range(1, numPins):
            pin = (oldPin + index) % numPins

            coord = coords[pin]

            xLine, yLine = linePixels(oldCoord, coord)

            # Fitness function
            lineSum = np.sum(imgMasked[yLine, xLine])

            if (lineSum > bestLine) and not(pin in previousPins):
                bestLine = lineSum
                bestPin = pin

        # Update previous pins
        if len(previousPins) >= minLoop:
            previousPins.pop(0)
        previousPins.append(bestPin)

        # Subtract new line from image
        lineMask = lineMask * 0
        cv2.line(lineMask, oldCoord, coords[bestPin], lineWeight, lineWidth)
        imgMasked = np.subtract(imgMasked, lineMask)

        # Save line to results
        lines.append((oldPin, bestPin))

        # plot results
        xLine, yLine = linePixels(coords[bestPin], coord)
        imgResult[yLine, xLine] = 0
        # cv2.imshow('image', imgResult)
        # cv2.waitKey(1)

        # Break if no lines possible
        if bestPin == oldPin:
            break

        # Prepare for next loop
        oldPin = bestPin

        # Print progress
        sys.stdout.write("\b\b")
        sys.stdout.write("\r")
        sys.stdout.write("[+] Computing line " + str(line + 1) + " of " + str(numLines) + " total")
        sys.stdout.flush()

    print("\n[+] Image threaded")

    # Wait for user and save before exit
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(savefolder, 'threaded.png'), imgResult)

    svg_output = open(os.path.join(savefolder, 'threaded.svg'), 'wb')
    header="""<?xml version="1.0" standalone="no"?>
    <svg width="%i" height="%i" version="1.1" xmlns="http://www.w3.org/2000/svg">
    """ % (width, height)
    footer="</svg>"
    svg_output.write(header.encode('utf8'))
    pather = lambda d : '<path d="%s" stroke="black" stroke-width="0.5" fill="none" />\n' % d
    pathstrings=[]
    pathstrings.append("M" + "%i %i" % coords[lines[0][0]] + " ")
    for l in lines:
        nn = coords[l[1]]
        pathstrings.append("L" + "%i %i" % nn + " ")
    pathstrings.append("Z")
    d = "".join(pathstrings)
    svg_output.write(pather(d).encode('utf8'))
    svg_output.write(footer.encode('utf8'))
    svg_output.close()

    csv_output = open(os.path.join(savefolder, 'threaded.csv'),'wb')
    csv_output.write("x1,y1,x2,y2\n".encode('utf8'))
    csver = lambda c1,c2 : "%i,%i" % c1 + "," + "%i,%i" % c2 + "\n"
    for l in lines:
        csv_output.write(csver(coords[l[0]],coords[l[1]]).encode('utf8'))
    csv_output.close()


if __name__=="__main__":
    print("[#1] Drawing image " + imgPath)
    training_image_to_draw()

    print("[#2] Convert to pins " + imgPath)
    convert_to_pins()

sys.exit()

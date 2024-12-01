# Video Analysis

# List of Assumption about the distance measure
# - in bottom 1/4 of the screen
# - contains a readable "m" or a 0 initial value
# - contains at least 2 characters
# - never moves right or left, up or down as the video runs
# - uses one colour, noticeable contrast with the background or text stroke
# - distance measure is always present

# Assumptions about the video
# - no skips greater than 1m


###########
# Note: As currently implemented, this script reads all the videos in files in it's directory
# The input is the set of videos, and the output is one graph per video showing the distance values of each frame
# To use for extracting frames based on distance, should save the values to a csv or other format in the code
# or, can refactor the code such that this is a library that can be called by other software


import cv2
import easyocr
import numpy as np
from time import perf_counter
import os
import matplotlib.pyplot as plt
import math

def initial_search(cam, reader, frame_num, amount_of_frames):
    found_dist = False

    ret, frame = cam.read()
    frame_num += 1

    bounds = None

    # initial search for textbox (may not be present frame 1)
    while(frame is not None):

        if(frame_num > amount_of_frames*0.5):
            return None, None, frame, frame_num

        frame_num += 1

        # Iterate through the frames of the video
        ret, frame = cam.read()

        # skip analyzing most of the frames
        # (for when distance measure is not present frame 1, speeds up the process)
        if(frame_num % 30 != 0):
            continue

        # Select only the bottom 1/4 of the frame
        frame = frame[int(len(frame)*3/4):]
        # Convert to greyscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize by a factor of 4
        frame = cv2.resize(frame, (len(frame[0])*4, len(frame)*4))

        # Read result from frame, only relevant chars
        result = reader.readtext(frame, allowlist='0123456789.:m ')

        # if no text found, go to next frame
        if(len(result) == 0):
            continue

        # Merge any seperated "m"s with previously read char sequence
        j = 0
        for (bbox, text, prob) in result:
            if(text == 'm' and j != 0):      
                result[j-1] = (result[j-1][0], result[j-1][1] + "m", result[j-1][2])
            j += 1


        # find the distance measure on the screen, must include 'm', and at least some 0's
        j = 0
        for (bbox, text, prob) in result:
            if(text == ''):
                continue
            if(text[-1] == 'm' and any(ch in text for ch in ["00", ".0", "0."])):
                if(all(c == '0' for c in result[j-1][1]) and 2.5 * bbox[0][0] - result[j-1][0][1][0] < 1.5 * bbox[1][0]):
                    bbox[0][0] = result[j-1][0][0][0]
                # define bounding box, length, width of distance measure
                bounds = [bbox[0][0], bbox[1][0] + 5, bbox[0][1], bbox[2][1]]
                # set num_frame to current frame in video
                found_dist = True
                break
            j += 1
        
        # all the below checks use the last text value from the reader, lowest, rightmost
        
        # if m is present, but initialization starts at non-zero, wait 20s then use it
        if(text[-1] == 'm' and len(text) > 2 and frame_num > 600):
            bounds = [bbox[0][0], bbox[1][0] + 5, bbox[0][1], bbox[2][1]]
            found_dist = True
            
        # break from loop if found the distance measure
        if(found_dist):
            break
        
        # no examples with m's, select last text box if 3 straight required chars present
        if(any(ch in text for ch in ["00.", "0.0", ".00", "000"])):
            bounds = [bbox[0][0], bbox[1][0] + 5, bbox[0][1], bbox[2][1]]
            # set num_frame to current frame in video
            found_dist = True
        
        # break from loop if found the distance measure
        if(found_dist):
            break

        # no examples with m's, select last two text boxes if meeting certain conditions
        if(len(result) > 1 and ((text == "0" and "00" in result[-2][1]) or (text == "00" and "0" in result[-2][1]))):
            # if large gap between this bounding box and previously read bounding box
            if(2.5 * bbox[0][0] - result[-2][0][1][0] < 1.5 * bbox[1][0]):
                bbox[0][0] = result[-2][0][0][0]
                bbox[0][1] = result[-2][0][0][1]
                bbox[2][1] = max(result[-2][0][0][1], bbox[2][1])
                bounds = [bbox[0][0], bbox[1][0] + 5, bbox[0][1], bbox[2][1]]
                text = text + "." + result[-2][1]
                # set num_frame to current frame in video
                found_dist = True
        
        # break from loop if found the distance measure
        if(found_dist):
            break

    return bounds, text, frame, frame_num

def check_bboxes(frame, bounds, reader, char_len):
    # grab just the portion of the image detected
    dist_frame = frame[bounds[2]:bounds[3], bounds[0]:bounds[1]]
    dist_frame = cv2.threshold(dist_frame, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

    # find all contours (enclosed areas)
    items = cv2.findContours(dist_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = items[0] if len(items) == 2 else items[1]


    bboxes = []

    for con in contours:
        # get dimensions of the contour, used for bounding box
        x,y,w,h = cv2.boundingRect(con)

        # TEST CODE: used to draw a grey box around the contour, show it on screen
        # temp = dist_frame.copy()
        # temp = cv2.rectangle(temp, (x, y), (x+w, y+h), (150, 0, 0), 2)
        # cv2.imshow('image', temp)
        # cv2.waitKey(0)

        # don't consider contour if not correct size
        if(h < len(dist_frame) * 0.40 or w > char_len * 3):
            continue
        # don't consider contour if not central vertically
        if(abs(y + h/2 - len(dist_frame)/2) > len(dist_frame)*0.06):
            continue
        # don't consider contour if connected to left or right border
        if(x+w > len(dist_frame[0]) - 1 or x < 2):
            continue
        # adjust x and y values to represent the bboxes in the full frame
        x1 = x + bounds[0]
        x2 = x + w + bounds[0]

        # never want any overlapping bounding boxes (ex: center hole of 0 vs main part)
        # select outermost bounding box, unless the two boxes have different horizontal centroids
        # if bbs have different centroids, keep only inner one
        delete_box = -1
        sub_box = True
        for i in range(len(bboxes)):
            inner = bboxes[i][0] < x1 and x2 < bboxes[i][1]
            outer = bboxes[i][0] >= x1 and bboxes[i][1] <= x2
            if(inner or outer):
                cent1 = int((x2 + x1)/2)
                cent2 = int((bboxes[i][1] + bboxes[i][0])/2)
                # unaligned, keep inner box
                if(abs(cent1 - cent2) > len(dist_frame)*0.05):
                    if(inner): delete_box = i
                    else: sub_box = False
                # aligned, keep outer box
                else:
                    if(inner): sub_box = False
                    else: delete_box = i

        if(sub_box):
            # create new bounding box
            bboxes.append([x1, x2, bounds[2], bounds[3]])
        if(delete_box >= 0):
            del bboxes[delete_box]
 
    # if one or less values found, mark as failed and move on
    if(len(bboxes) < 2):
        return [], -1

    # sort by location of left edge
    bboxes = sorted(bboxes, key=lambda x: x[0])

    # get list of centroids, gaps between centroids of the bboxes
    centroids = [int((bb[1] + bb[0])/2) for bb in bboxes]
    widths = [int((bb[1] - bb[0])/2) for bb in bboxes]

    size_hchar = max(widths)

    # get size of char by using the gap sizes
    # picks a median gap size value to avoid using the gap involving a . (ex: 0.0 vs 00)
    
    i = 0
    # create new bounding boxes using the centroids and char lengths
    while(i < len(centroids)):
        bboxes[i][0] = centroids[i] - size_hchar - 5
        bboxes[i][1] = centroids[i] + size_hchar + 5
        # check to make sure a numeric char is read from the bounding box correctly
        # x is image as is, x1 is thresholded image (easier to read)
        x = frame[bboxes[i][2]:bboxes[i][3], bboxes[i][0]:bboxes[i][1]]
        x1 = cv2.threshold(x, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
        checker = reader.recognize(x, allowlist='0123456789')
        checker2 = reader.recognize(x1, allowlist='0123456789')
        # confidence of the OCR system that the char is present, must be confident
        if(checker[0][2] < 0.60 and checker2[0][2] < 0.70):
            del bboxes[i]
            del centroids[i]
        else:
            i += 1
    
    if(len(centroids) < 2):
        return [], -1
    
    # Deal with leftmost bounding box
    x = frame[bboxes[0][2]:bboxes[0][3], bboxes[0][0]:bboxes[0][1]]
    x1 = cv2.threshold(x, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    if(max(x1[:, 0]) == min(x1[:, 0])):
        color = x1[0][0]
        count_border = 0
        count_not = 0
        for i in range(int(len(x1[0])/2)):
            if(min(x1[:, i]) == color and max(x1[:, i]) == color):
                count_border += 1
            elif(min(x1[:, i]) == 255-color and max(x1[:, i]) == 255-color):
                break
            else:
                count_not += 1
                if(count_not > 3):
                    count_border = 0
                    break
        if(count_border > 0):
            bboxes[0][0] += count_border
            bboxes[0][1] += count_border
    
    gaps = [centroids[i+1] - centroids[i] for i in range(len(centroids) - 1)]
    ind = gaps.index(max(gaps))
    
    return bboxes, ind+1


# creates image reader
reader = easyocr.Reader(['en']) # specify the language 


# files in directory
files = os.listdir()

for f_name in files:
    # open video
    if(".py" in f_name or "." not in f_name):
        continue

    ### for checking an individual video ###
    # if(f_name != 'STMH55-STMH65_Street A_221107_103739.mpg'):
    #     continue

    cam = cv2.VideoCapture(f_name)


    amount_of_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cam.get(cv2.CAP_PROP_FPS)

    bounds = []

    ret, frame = cam.read()

    frame_num = 0

    bboxes = []
    text = "a"

    # keep trying the "find text" -> "separate into bounding boxes" process
    # until successful (text resembling distance is found, bounding boxes match text)
    while(len(bboxes) < len(text)):

        bounds, text, frame, frame_num = initial_search(cam, reader, frame_num, amount_of_frames)
        if(bounds is None):
            print("Failed Distance Read. File: " + f_name)
            break


        # add additional length to bounding box in case early chars were missed by the reader
        length = bounds[1] - bounds[0]
        length_added = round(length * 2 / len(text))

        char_len = round(length / (len(text) + 2))

        bounds[0] = max(bounds[0] - length_added, 0)

        text = text.replace(".", "")
        text = text.replace(" ", "")
        text = text.replace("m", "")

        bboxes, dec_ind = check_bboxes(frame, bounds, reader, char_len)
    
    if(bounds is None):
        continue

    bounds = [bboxes[0][0], bboxes[-1][1], bboxes[0][2], bboxes[0][3]]

    frame_labels = []

    # probability adjustments
    rough_odds = {-1: {0: 0.55, 1:0.40, 2:0.04}, 0: {0: 0.90, 1:0.80}}

    # Go through all remaining frames, recording distances for every 10 frames

    nine_count = 0

    prev_num = [0] * len(bboxes)
    prev_nine1 = [0] * 9
    prev_nine01 = [0] * 9

    while(True):
        ret, frame = cam.read()
        frame_num += 1

        # once at end of video, finish loop
        if(frame is None):
            break
        
        # only check every 10 frames
        if(frame_num % 10 != 0):
            continue

        if(frame_num in [26300, 26310]):
            check = 0


        # Select only the bottom 1/4 of the frame
        frame = frame[int(len(frame)*3/4):]
        # Convert to greyscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize by a factor of 4
        frame = cv2.resize(frame, (len(frame[0])*4, len(frame)*4))

        # if only 9s in non-decimal digits, need to check for new digits (9->10, 99->100)
        if(nine_count > 3):
            # extend the bounds in both directions for any new digits
            temp_bounds = [max(bounds[0] - length_added, 0), min(bounds[1] + length_added, len(frame[0]) - 1), bounds[2], bounds[3]]

            # read text to check if transitioned (9 -> 10, 99 -> 100)
            full_read = reader.readtext(frame[temp_bounds[2]:temp_bounds[3], temp_bounds[0]:bounds[1]], allowlist='0123456789.:m ')
            full_text = ""
            for (bbox, text, prob) in full_read:
                full_text += text
            full_text = full_text.replace(".", "")
            # if text starts with a 1, or 2nd digit is a 0, do a more in depth check
            if((len(full_text) > 0 and full_text[0] == '1') or (len(full_text) > 1 and full_text[1] == '0')):
                temp_bboxes, temp_dec_ind = check_bboxes(frame, temp_bounds, reader, char_len)
            else:
                temp_bboxes = []
            # if one new bounding box (for new digit) update bounding boxes, location of decimal, prev number array
            if(len(temp_bboxes) == len(bboxes) + 1):
                bboxes = temp_bboxes
                dec_ind = temp_dec_ind
                nine_count = 0
                prev_num.insert(0, 0.0)


        read_number = 0
        # check all the bounding boxes
        for i, bb in enumerate(bboxes):
            num = frame[bb[2]:bb[3], bb[0]:bb[1]]
            num1 = cv2.threshold(num, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
            digit = -1 * i + dec_ind - 1
            # if digit is 100s, 10s, rarely needs to change, only check if next number is 9
            if(digit > 0):
                if(prev_num[i+1] == 9):
                    char_list = str(int(prev_num[i])) + str(int(prev_num[i]+1)%10)
                    checker = reader.recognize(num1, allowlist=char_list)
                    if(checker[0][2] < 0.80):
                        checker2 = reader.recognize(num1, allowlist=char_list.replace(checker[0][1], ''))
                        if(checker2[0][2] != 0 or checker[0][2] == 0):
                            read_number += prev_num[i] * pow(10, digit)
                            continue
                    
                    number = float(checker[0][1])
                    number = (number % 10) % 100
                    read_number += float(number) * pow(10, digit)
                    prev_num[i] = float(number)
                else:
                    read_number += prev_num[i] * pow(10, digit)
                
                continue

            if(digit < -1):
                checker = reader.recognize(num1, allowlist='0123456789')
                # if OCR system is unsure, just use value from previous frame
                if(checker[0][2] < 0.80):
                    read_number += prev_num[i] * pow(10, digit)
                else:
                    read_number += float(checker[0][1]) * pow(10, digit)
                    prev_num[i] = float(checker[0][1])
                continue
            
            # digit is 1 or 0.1 digit, significant but fast changing, needs more analysis
            checker = reader.recognize(num1, allowlist='0123456789')
            if(checker[0][2] < 0.80):
                read_number += prev_num[i] * pow(10, digit)
            else:
                if(digit == 0):
                    # ones digit is mostly stable, use mode of previous 9 readings for comparison
                    check_num = max(set(prev_nine1), key=prev_nine1.count)
                else:
                    # 0.1s digit is not stable, use previous reading for comparison
                    check_num = max(set(prev_nine01), key=prev_nine01.count)
                    if(prev_nine01.count(check_num) < 5):
                        check_num = prev_num[i]

                # if new digit is not the same or one greater than last digit
                if(float(checker[0][1]) not in [check_num, (check_num+1)%10]):
                    # get list of probabilities for most likely digits
                    prob_list = [[checker[0][1], checker[0][2]]]
                    tot_prob = checker[0][2]
                    chars = '0123456789'.replace(checker[0][1], '')
                    while(tot_prob < 0.995):
                        checker2 = reader.recognize(num1, allowlist=chars)
                        if(checker2[0][1] == ''):
                            break
                        prob_list.append([checker2[0][1], checker2[0][2] * (1-tot_prob)])
                        tot_prob += checker2[0][2] * (1-tot_prob)
                        # reader returns most likely char, so remove it from read list and run again
                        # to get more digits
                        chars = chars.replace(checker2[0][1], '')
                        if(len(chars) == 0):
                            break
                    
                    # adjust probabilties given the previous value
                    for value in prob_list:
                        # difference between previous and current
                        diff = (int(value[0]) - check_num) % 10
                        if(diff in rough_odds[digit]):
                            value[1] *= rough_odds[digit][diff]
                            # additional adjustment, if lower digit of previous is a 9 more likely to flip
                            if(i+1 < len(prev_num) and prev_num[i+1] == 9 and diff == 1):
                                value[1] *= 1.5
                        else:
                            value[1] *= 0.01
                    
                    best_guess = sorted(prob_list, key=lambda x: x[1])[-1][0]
                    number = float(best_guess)
                    number = (number % 10) % 100
                    read_number += float(number) * pow(10, digit)
                    prev_num[i] = float(number)
                    if(digit == 0):
                        prev_nine1.append(float(number))
                        del prev_nine1[0]
                    else:
                        prev_nine01.append(float(number))
                        del prev_nine01[0]
                # new digit is the same or one more than previous digit, update as normal
                else:
                    read_number += float(checker[0][1]) * pow(10, digit)
                    prev_num[i] = float(checker[0][1])
                    if(digit == 0):
                        prev_nine1.append(float(checker[0][1]))
                        del prev_nine1[0]
                    else:
                        prev_nine01.append(float(checker[0][1]))
                        del prev_nine01[0]
        
        frame_labels.append((read_number, frame_num))

        nine = True
        for number in prev_num[:dec_ind]:
            if(number != 9):
                nine = False
        if(nine):
            nine_count += 1
        else:
            nine_count = max(nine_count - 1, 0)


    
    # Smoothing
    gap_sizes = []
    len_gaps = []
    dist_covered = []
    gap_comps = []
    for i in range(1, len(frame_labels) - 1):

        # check the distance between frames, if large there's probably an issue to fix
        diff = frame_labels[i+1][0] - frame_labels[i][0]
        if(diff < -0.3 or diff > 0.5):
            match_val = -1
            # check distance between previous two frames, to guess how much this gap "should've" been
            diff2 = frame_labels[i][0] - frame_labels[i-1][0]
            # if previous gap was normal, remove the expected gap size by the actual gap size
            if(diff2 > -0.3 and diff2 < 0.5):
                diff -= diff2
            
            # search through all gaps previously found in the timeline, look for a match
            # Rationale: if the frame reader jumped +10m by reading a wrong char, it should jump -10m at some point
            for j in range(len(gap_sizes)):
                # if corresponding gaps seem to match (10.1 and -9.8 are close, likely match)
                if(abs(gap_sizes[j] + diff) < 1.0):
                    match_val = j
                    break
            
            deleted = False
            # if found a match, adjust all frames in between these two jumps 
            # (i.e. add the size of the jump back to the correct value)
            if(match_val >= 0):
                for k in range(len_gaps[j]):
                    frame_labels[i-k] = (frame_labels[i-k][0] + diff, frame_labels[i-k][1])
                # remove this jump, as well as any jumps that occured after it
                # Assumption is that jumps are LIFO, get undone in opposite order,
                # so any jumps above in the stack are assumed to be correct readings
                del gap_sizes[:j+1]
                del len_gaps[:j+1]
                del gap_comps[:j+1]
                del dist_covered[:j+1]
                deleted = True
            # if there's multiple jumps present, look if this jump matches the sum of two consecutive previous jumps
            # Rationale: if readers makes two mistakes, then fixes them both at once
            elif(len(gap_sizes) > 1):
                match_val = -1
                # if two consecutive previous jumps match current jump
                for j in range(len(gap_sizes) - 1):
                    if(abs(gap_sizes[j] + gap_sizes[j+1] + diff) < 1.0):
                        match_val = j
                        break
                # make two adjustments, one for each jump, over the region they affected
                if(match_val >= 0):
                    for k in range(len_gaps[j]):
                        frame_labels[i-k] = (frame_labels[i-k][0] - gap_sizes[j], frame_labels[i-k][1])
                    for k in range(len_gaps[j+1]):
                        frame_labels[i-k] = (frame_labels[i-k][0] - gap_sizes[j+1], frame_labels[i-k][1])
                    # remove both jumps, and any jumps that happened after these two
                    del gap_sizes[:j+2]
                    del len_gaps[:j+2]
                    del gap_comps[:j+2]
                    del dist_covered[:j+2]
                    deleted = True
            # if jump is not associated with any other jumps, add it to the list of gaps
            if(not deleted):
                # distance of the gap in metres
                gap_sizes.insert(0, diff)

                # counter for how many frames are affected by the jump
                len_gaps.insert(0, 0)

                # maximum distance the gap can affect
                # if the gap involves the 10s digit (say, 29.9m -> 10.0m) it should hopefully be fixed by the next 10s digit change
                # so, round down to the nearest order of magnitude (10.3 -> 10, 8.3 -> 1)
                # *1.1 allows some tolerance for the order of magnitude (9.98 -> 10), * 2 gives some tolerance for the distance
                gap_comps.insert(0, pow(10, round(math.log(abs(diff*1.1), 10) - 0.5)) * 2)

                # how much distance has been covered since jump was added
                dist_covered.insert(0, 0)

        if(len(len_gaps) > 0):
            for j in range(len(len_gaps)):
                len_gaps[j] += 1
            if(diff > -0.3 and diff < 0.5):
                # if not currently dealing with a new gap, increase all distance covered values by the distance covered
                for j in range(len(len_gaps)):
                    dist_covered[j] += abs(diff)
                # if the distance since the gap was added exceeds the computed threshold, remove the gap from the list
                if(dist_covered[-1] > gap_comps[-1]):
                    del len_gaps[-1]
                    del gap_sizes[-1]
                    del gap_comps[-1]
                    del dist_covered[-1]



    x = [z[1] for z in frame_labels]
    y = [z[0] for z in frame_labels]
    plt.plot(x, y)
    plt.savefig('imgs/fig_' + f_name[:-4] + ".png")
    plt.show(block=True)
    
    check = 0

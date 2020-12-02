import numpy as np
import cv2
import os

from functions_characters import fix_i_j

def get_characters(raw_image,max_line_height,line,word):

    # === Find Contours

    mo_image = raw_image.copy()
    contour0 = cv2.findContours(mo_image.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(cnt,2,True) for cnt in contour0[0]]

    # === Extract Bounding Rectangles
    maxArea = 0
    rect=[]
    for ctr in contours:
        maxArea = max(maxArea,cv2.contourArea(ctr))

    areaRatio = 0.008

    for ctr in contours:
        if cv2.contourArea(ctr) > maxArea * areaRatio: 
            rect.append(cv2.boundingRect(cv2.approxPolyDP(ctr,1,True)))
            
    #Find max width of contour
    max_w = 0
    
    for i in rect:
        x = i[0]
        y = i[1]
        w = i[2]
        h = i[3]

        if(w>max_w):
            max_w = w

    rect.sort(key=lambda b: b[0])

    #Fix i and j
    rect = fix_i_j(rect, max_line_height, max_w)
    
    # remove artifacts -  0 height by the i&j dot fixing functions
    minus_count = 0
    minus_list = []
    for i in rect:
        x = i[0]
        y = i[1]
        w = i[2]
        h = i[3]
        
        if h<0:
            minus_list.append(minus_count)
        
        minus_count = minus_count + 1
    
    rect = np.delete(rect, minus_list, axis=0)
    
    rect_segmented_image = mo_image.copy()
    
    all_letters = []

    count = 0
    
    path = os.getcwd();
    
    for i in rect:
        x = i[0]
        y = i[1]
        w = i[2]
        h = i[3]
        
        p1 = (x,y)
        p2 = (x+w,y+h)
        
        letter = mo_image[int(y):int(y+h),int(x):int(x+w)]
        
        #resize letter image to 32x32 
        #resize letter content to 28x28
        
        o_height = letter.shape[0]
        o_width = letter.shape[1]
        
        #if errors occurs due to the unwanted artifacts, then the height will somehow become zero.
        if (o_height == 0):
            letter = np.zeros((30, 30, 1), np.uint8)
            o_height = letter.shape[0]
            o_width = letter.shape[1]
        
        #resize height to 28 pixels
        #we need three different conditions to work well with the aspect ratios
        
        if(o_height>o_width): # height greater than width

            aspectRatio = o_width / (o_height*1.0)
            
            height = 26
            width = int(height * aspectRatio)
            letter = cv2.resize(letter, (width,height))
            
            #add border which results adding of padding
            
            remaining_pixels_w = abs(32 - letter.shape[1])
            add_left = int(remaining_pixels_w / 2)
            add_right = remaining_pixels_w - add_left
            letter = cv2.copyMakeBorder(letter, 0, 0, add_left, add_right, cv2.BORDER_CONSTANT, value=(0,0,0))
            
            remaining_pixels_h = abs(32 - letter.shape[0])
            add_top = int(remaining_pixels_h / 2)
            add_bottom = remaining_pixels_h - add_top
            letter = cv2.copyMakeBorder(letter, add_top, add_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            
            # =================
            
        elif(o_width>o_height): # width greater than height
            
            aspectRatio = o_height / (o_width*1.0)
            
            width = 26
            height = int(width * aspectRatio)
            
            letter = cv2.resize(letter, (width,height))
            
            #add border which results adding of padding
            remaining_pixels_w = abs(32 - letter.shape[1])
            add_left = int(remaining_pixels_w / 2)
            add_right = remaining_pixels_w - add_left
            letter = cv2.copyMakeBorder(letter, 0, 0, add_left, add_right, cv2.BORDER_CONSTANT, value=(0,0,0))
            
            remaining_pixels_h = abs(32 - letter.shape[0])
            add_top = int(remaining_pixels_h / 2)
            add_bottom = remaining_pixels_h - add_top
            letter = cv2.copyMakeBorder(letter, add_top, add_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            
            # =================
        
        else: # both height and width equal
            letter = cv2.resize(letter, (26,26))
            
            #add border which results adding of padding
            remaining_pixels_w = abs(32 - letter.shape[1])
            add_left = int(remaining_pixels_w / 2)
            add_right = remaining_pixels_w - add_left
            letter = cv2.copyMakeBorder(letter, 0, 0, add_left, add_right, cv2.BORDER_CONSTANT, value=(0,0,0))
            
            remaining_pixels_h = abs(32 - letter.shape[0])
            add_top = int(remaining_pixels_h / 2)
            add_bottom = remaining_pixels_h - add_top
            letter = cv2.copyMakeBorder(letter, add_top, add_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            
            # =================
        
        
        #uncomment to get letter images
        
        #cv2.imwrite(os.path.join(path,'letters','line_'+str(line)+'word_'+str(word)+'letter_'+str(count)+'.png'),letter)
        
        count = count + 1
        #cv2.imshow('segmented_letter.png',letter)
        letter = letter / 255.0
        
        letter = np.reshape(letter,(1024,1))
        
        all_letters.append(letter)

        #=================================

        cv2.rectangle(rect_segmented_image,(int(x),int(y)),(int(x+w),int(y+h)),255,1)
        
    cv2.imwrite(os.path.join(path,'words','line_'+str(line)+'word_'+str(word)+'.png'), rect_segmented_image)
    print(count)
    return all_letters
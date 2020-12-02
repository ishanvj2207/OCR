#The starting point of the backend works
import cv2

from segmentation_words import get_words
from segmentation_characters import get_characters
#input image should have white bg, black text

def perform_ocr(img_url):    
    raw_image = cv2.imread(img_url,0)
    #cv2.imshow('image',img_url)
    
        #get all the words (as an numpy image array), words on each line, and maximum height on that line
    all_words, words_on_line, max_height_on_line = get_words(raw_image)
    
    print ("no. of lines = ",len(words_on_line))
    print (words_on_line)
    print ("no. of words = ",len(all_words))
    
    count = 0
    for i in range(0, len(words_on_line)):
        for j in range(0, words_on_line[i]):
            all_characters = get_characters(all_words[count],max_height_on_line[i],i,j)
            #cv2.imwrite("all_words "+str(count)+'.png',all_words[count])
            
            cv2.waitKey()            
            count = count + 1
            
        
    
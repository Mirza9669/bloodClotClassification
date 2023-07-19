import os
import cv2
import numpy as np


folder = 'D:\\FYP\\mayo-clinic-strip-ai\\test'

    
for file in os.listdir(folder):
    print(folder + '\\' + file)
    img = cv2.imread(folder + "\\" + file)
    img
    name, ext = file.split(".")
    # convert img to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # otsu threshold
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU )[1] 

    # apply morphology open
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17,17))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    morph = 255 - morph

    # find contours and bounding boxes
    bboxes = []
    bboxes_img = img.copy()
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        cv2.rectangle(bboxes_img, (x, y), (x+w, y+h), (0, 0, 0), 1)
        bboxes.append((x,y,w,h))

    # sort bboxes on x coordinate
    def takeFirst(elem):
        return elem[2]

    bboxes.sort(key=takeFirst)
    print(bboxes)
    bbox = bboxes[-1]
    print(bbox[0])
    print(bbox[1])


    # if bbox[0] == 0 or bbox[1] == 0:
    #     bbox = bboxes[-2]
    #     print(bbox)
        

    # Calculate the center of the largest bounding box
    max_center = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)

    # Select bounding boxes within a certain distance from the largest bounding box
    distance_threshold = 10000
    selected_bboxes = []
    for bbox in bboxes:
        center = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
        distance = np.sqrt((center[0] - max_center[0])**2 + (center[1] - max_center[1])**2)
        print("Distance: ",distance)
        if distance <= distance_threshold and bbox[2] >= 1500:
            selected_bboxes.append(bbox)
        
    # print("Selected Boxes: ",selected_bboxes)
        
    def takeFirstselected(elem):
        return elem[0]

    selected_bboxes.sort(key=takeFirstselected)

    print("Selected Boxes: ",selected_bboxes)

    '''
    final_bboxes = []
    for bbox in selected_bboxes:
        x,y,w,h = cv2.boundingRect(bbox)
        cv2.rectangle(bboxes_img, (x, y), (x+w, y+h), (0, 0, 0), 1)
        final_bboxes.append((x,y,w,h))
    '''    

    def calculate_combined_dimensions(boxes):
        min_x = min(box[0] for box in boxes)
        max_x = max(box[0] + box[2] for box in boxes)
        min_y = min(box[1] for box in boxes)
        max_y = max(box[1] + box[3] for box in boxes)

        combined_width = max_x - min_x
        combined_height = max_y - min_y

        return min_x, min_y, combined_width, combined_height

    # Example usage
    final_box = []
    x,y,combined_width, combined_height = calculate_combined_dimensions(selected_bboxes)
    final_box = (x,y,combined_width,combined_height)
    print(final_box)
    print("Combined Width:", combined_width)
    print("Combined Height:", combined_height)
    cv2.rectangle(bboxes_img, (x, y), (x+combined_width, y+combined_height), (0, 0, 0), 1)
    # final_box.append((x,y,w,h))    
    cv2.imwrite("D:\Images\Multiple Clot and Bboxes\ "+ str(name) +"-BBoxes.png",bboxes_img)
    bbox = (x,y,combined_width,combined_height)
    print(bbox)
    # get largest width of bboxes
    maxwidth = bbox[-2]

    # stack cropped boxes with 10 pixels padding all around
    result = np.full((1,maxwidth+20,3), (255,255,255), dtype=np.uint8)
    # for bbox in bboxes:
    (x,y,w,h) = bbox
    crop = img[y-10:y+h+10, x-10:x+maxwidth+10] 
    # crop_resized = cv2.resize(crop, (maxwidth, 100))
    # crop_padded = cv2.copyMakeBorder(crop_resized, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    crop_height, crop_width, channels = crop.shape
    result_height, result_width, channels = result.shape
    if result_width != crop_width:
        if result_width-crop_width < 100 and result_width-crop_width > 0 :
            difference = result_width-crop_width
            result =  np.full((1,maxwidth+20-difference,3), (255,255,255), dtype=np.uint8)
        
        elif crop_width-result_width < 100 and crop_width-result_width > 0 :
            difference = result_width-crop_width
            result =  np.full((1,maxwidth+20+difference,3), (255,255,255), dtype=np.uint8)
        
    
    result = np.vstack((result, crop))    


    # save result
    # cv2.imwrite("D:\Images\Multiple Clot and Bboxes\ "+ str(name) +"-BBoxes.png",bboxes_img)

    cv2.imwrite("F:\\Evaluation\\test\\"+ str(name) +".png",result)
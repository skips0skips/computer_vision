import glob, os
i=0
j=0
for i in range(965):
    if i % 2 == 0:
        os.remove(f'iphone\image_l\{"{:000006}".format(i)}.jpg')
        #print(i)
    else:
        os.rename(f'iphone\image_l\{"{:000006}".format(i)}.jpg', f'iphone\image_l\{"{:000006}".format(j)}.jpg')
        j+=1
        print(j)

# import cv2
# from tqdm import tqdm
# capture = cv2.VideoCapture(r'iphone_2\video_2023-02-18_15-47-33.mp4')
# frameNr = 0
# while (True):

#     success, frame = capture.read()

#     if success:
#         cv2.imwrite(f'iphone_2\image_l\{"{:000006}".format(frameNr)}.jpg', frame)
#     else:
#         break
#     print(frameNr)
#     frameNr = frameNr+1
 
# capture.release()
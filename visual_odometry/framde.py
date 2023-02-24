import cv2
from os import listdir, remove
from os.path import isfile, join


def framde():
    path_image = r'data\image_l' # путь до папки где хранятся фото с прошлого запуска
    for f in listdir(path_image):
        remove(join(path_image, f)) # удаляем все прошлые фото 

    path_video = r'data\video_i' # путь до папки где хранятся видео
    onlyfiles = [f for f in listdir(path_video) if isfile(join(path_video, f))] #массив ссылок до файлов в папке
    frameNr = 0 # номер изображения
    for i in range(len(onlyfiles)):
        print(f"предобработка видео: {onlyfiles[i]}")
        capture = cv2.VideoCapture(f'data\\video_i\\' + onlyfiles[i])#
        while (True):
            success, frame = capture.read()
            if success:
                cv2.imwrite(f'data\\image_l\\{"{:000006}".format(frameNr)}.jpg', frame)
            else:
                break
            #print(frameNr)
            frameNr = frameNr+1
        capture.release()
    print('Нарезка фреймов завершена')
    pass


# import glob, os функция для удаления лишних фреймов 
# i=0
# j=0
# for i in range(965):
#     if i % 2 == 0:
#         os.remove(f'iphone\image_l\{"{:000006}".format(i)}.jpg')
#         #print(i)
#     else:
#         os.rename(f'iphone\image_l\{"{:000006}".format(i)}.jpg', f'iphone\image_l\{"{:000006}".format(j)}.jpg')
#         j+=1
#         print(j)
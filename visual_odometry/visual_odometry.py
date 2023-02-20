import os
import numpy as np
import cv2

from lib.visualization import plotting
from lib.visualization.video import play_trip

from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

class VisualOdometry():
    def __init__(self, data_dir):
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        self.gt_poses = self._load_poses(os.path.join(data_dir,"poses.txt"))
        self.images = self._load_images(os.path.join(data_dir,"image_l"))
        self.orb = cv2.ORB_create(99999999) #3000     #инициализация детектора ORB 7500 - хорошо
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    @staticmethod
    def _load_calib(filepath):
        """
        Функция загружает калибровку камеры
        Параметры:
        filepath - путь к файлу камеры (str)
        Возвращает:
        K (ndarray): Внутренние параметры
        P (ndarray): Проекционная матрица

        """
        with open(filepath, 'r') as f:   #открывается файл в режиме чтения
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')     #params одномерный массив, который заполнен данными из одной целой строки файла
            P = np.reshape(params, (3, 4))      #изменяет форму массива на размер (3, 4)
            K = P[0:3, 0:3]     #уберает нули
        return K, P

    @staticmethod
    def _load_poses(filepath):
        """
        Функция загружает позиции GT
        Параметры:
        filepath - путь к файлу камеры (str)
        Возвращает:
        poses (ndarray):  массив с позицией
        """
        poses = []
        with open(filepath, 'r') as f:      #открывается файл в режиме чтения
            for line in f.readlines():      #считывается строка
                T = np.fromstring(line, dtype=np.float64, sep=' ')      #размерность [0:12]
                T = T.reshape(3, 4)     #размерность [3:4]
                T = np.vstack((T, [0, 0, 0, 1]))    #добавляет массив [0,0,0,1] в конец массива
                poses.append(T)
        return poses

    @staticmethod
    def _load_images(filepath):
        """
        Функция загружает изображения
        Параметры:
        filepath - путь к каталогу изображений (str)
        Возвращает:
        images (list): изображения в оттенках серого
        Описание:
        1)в лист image_paths по пути filepath добавляется сортированный список , содержащий имена файлов и директорий в каталоге
        2)изображения из листа image_paths преобразуются в массив ndarray с оттенком серого и добавляются в лист
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    @staticmethod
    def _form_transf(R, t):
        """
        Функция создает матрицу преобразования из заданной матрицы поворота и вектора перемещения
        Параметры:
        R (ndarray): Матрица вращения
        t (list): Вектор перевода
        Возвращает:
        T (ndarray): Матрица преобразования
        """
        # двумерный массив у которого все элементы по диагонали равны 1, а все остальные равны 0
        T = np.eye(4, dtype=np.float64) # 4 -количество строк выходного массива
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, i):
        """
        Эта функция обнаруживает и вычисляет ключевые точки, дескриптор из i-1-го и i изображений, используя класс или объект
        Параметры:
        i - Текущий кадр (int)
        Возвращает:
        q1 (ndarray): Хорошие ключевые точки соответствия положения на i-1-м изображении
        q2 (ndarray): Хорошие ключевые точки соответствия положения на i изображении
        """
        # Нахождение ключевых точек и дескрипторов с помощью ORB
        kp1, des1 = self.orb.detectAndCompute(self.images[i - 1], None)
        kp2, des2 = self.orb.detectAndCompute(self.images[i], None)
        # Нахождение совпадений, где k число лучших совпадений для каждого дескриптора
        matches = self.flann.knnMatch(des1, des2, k=2)

        # Нахождение совпадений, которые не имеют большого расстояния
        good = []
        try:
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        draw_params = dict(matchColor = -1, # обрисовкасоединяющих линий зеленым цветом
                 singlePointColor = None, #Цвет отдельных ключевых точек (кружков), что означает, что ключевые точки не имеют совпадений
                 matchesMask = None, # Маска, определяющая, какие совпадения будут нарисованы. Если маска пуста, все совпадения отображаются
                 flags = 2) #Флаги, устанавливающие функции рисования

        # Рисует найденные совпадения ключевых точек из двух изображений.
        # images[i], images[i-1] - первое и второе исходное изображение
        # kp1, kp2 - ключевые точки из первого и второго исходного изображения
        # good - список точек соответствия первого и воторого изображения
        # outImg - вывод изображения
        img3 = cv2.drawMatches(self.images[i], kp1, self.images[i-1],kp2, good ,outImg = None,**draw_params)

        # Меняет размер drawMatches на размер одного изображения
        height, width = self.images[i].shape
        img3 = cv2.resize(img3, (width, height))

        # cv2.imshow("image", img3)
        # cv2.destroyWindow("image")
        # cv2.waitKey(200)

        # Получение списока точек соответствия первого и воторого изображения
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return q1, q2

    def get_pose(self, q1, q2):
        """
        Функция вычисляет матрицу преобразования
        Параметры:
        q1 (ndarray): Хорошие ключевые точки соответствия положения на i-1-м изображении
        q2 (ndarray): Хорошие ключевые точки соответствия положения на i изображении
        Возращает:
        transformation_matrix (ndarray) - Матрица преобразования
        """
        # Вычисляет существенную матрицу из  точек на двух изображениях.
        #threshold - максимальное расстояние от точки до эпиполярной линии в пикселях, за пределами которого точка считается выбросом
        E, _ = cv2.findEssentialMat(q1, q2, self.K, threshold=1)

        # Разложение существенной матрицы на ветор перемещения и матрицу поворота
        R, t = self.decomp_essential_mat(E, q1, q2)

        # Получаем матрицу преобразования
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        return transformation_matrix

    def decomp_essential_mat(self, E, q1, q2):
        """
        Функция разложиения существенной матрицы
        Параметры:
        E (ndarray): Существенная матрица
        q1 (ndarray): Хорошие ключевые точки соответствия положения на i-1-м изображении
        q2 (ndarray): Хорошие ключевые точки соответствия положения на i изображении
        Возращает:
        right_pair (list): Содержит матрицу поворота и вектор перемещения
        """
        def sum_z_cal_relative_scale(R, t):
            # Получение матрицы преобразования
            T = self._form_transf(R, t)
            # Создание проекционной матрицы
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

            # Триангуляция 3D-точек
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            hom_Q2 = np.matmul(T, hom_Q1)

            # Не Гомогенизировать
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            # Поиск количества точек, имеющих положительную координату z в обеих камерах
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

            # Формирование пары точек и вычисление относительного масштаба
            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
                                     np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        # разложиение существенной матрицы
        # R1, R2 - первая и вторая из возможных матриц вращения
        # t - один из возможных вариантов перевода
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t) #удаляется лишняя ось в массиве

        # Составление списока различных возможных пар
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        # Определение правильного решения
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Отбор пары, в которой больше всего точек с положительной координатой z
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale

        return [R1, t]


def main():
    data_dir = 'train_data'#"KITTI_sequence_1"  # Try KITTI_sequence_2 too
    vo = VisualOdometry(data_dir)

    # play_trip(vo.images)  # Прокомментируйте, чтобы не воспроизводить поездку

    gt_path = []

    estimated_path = []
    for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="pose")): #передается массив с позицией, i - индекс
        if i == 0:
            cur_pose = gt_pose
        else:
            q1, q2 = vo.get_matches(i)
            transf = vo.get_pose(q1, q2) #transf = np.nan_to_num(transf, neginf=0,posinf=0)
            transf = np.nan_to_num(transf, neginf=0,posinf=0)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf)) #вычисляем обратную матрицу transf и находим произведение с cur_pose
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
    # Отрисовка графиков
    # plotting.visualize_paths(estimated_path, estimated_path, "Visual Odometry", file_out=os.path.basename(data_dir) + ".html")

    x = []
    y = []
    z = []
    for i in estimated_path:
        x.append(i[0])
        y.append(i[1])
        z.append(0)

    t = np.linspace(0, 3, len(x))
    dataSet = np.array([x, y, z])  # Комбинируем наши позиционные координаты
    numDataPoints = len(t)

    def animate_func(num):
        ax.clear()  # Очищаем фигуру для обновления линии, точки,
                    # заголовка и осей  # Обновляем линию траектории (num+1 из-за индексации Python)
        if (num < len(x)):
            ax.plot3D(dataSet[0, :num+1], dataSet[1, :num+1],
                        dataSet[2, :num+1], c='blue')    # Обновляем локацию точки
            ax.scatter(dataSet[0, num], dataSet[1, num], dataSet[2, num],
                        c='blue', marker='o')    # Добавляем постоянную начальную точку
            ax.plot3D(dataSet[0, 0], dataSet[1, 0], dataSet[2, 0],
                        c='black', marker='o')    # Задаем пределы для осей
        ax.set_xlim3d([min(x)-5, max(x)+10])
        ax.set_ylim3d([min(y)-5, max(y)+10])
        ax.set_zlim3d([0, 1])
        # ax.set_axis_off()
        # ax.view_init(90, 270)


        # Добавляем метки
        # ax.set_title('Траектория движения \nTime = ' + str(np.round(t[num], decimals=2)) + ' sec')
        ax.set_title('Траектория движения')

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    line_ani = animation.FuncAnimation(fig, animate_func, interval=1/len(x),
                                    frames=numDataPoints)
    # plt.show()

    cv2.destroyAllWindows()
    plt.close('all')
    f = r"C:/Users/Hp/Desktop/Projects/computer_vision/visual_odometry/train_data/вид_сбоку.gif"
    # f = r"C:/Users/Hp/Desktop/Projects/computer_vision/visual_odometry/train_data/вид_сверху.gif"
    writergif = animation.PillowWriter(fps = numDataPoints/6)
    line_ani.save(f, writer=writergif)

if __name__ == "__main__":
    main()

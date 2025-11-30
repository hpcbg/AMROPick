"""
OAK-D Camera Module
Клас за замяна на cv2.VideoCapture с OAK-D камера функционалност
"""

import depthai as dai
import cv2
import numpy as np


class OAKDCamera:
    """Клас за замяна на cv2.VideoCapture с OAK-D камера"""

    def __init__(self, rgb_resolution="4K", fps=30, preview_size=(1920, 1080)):
        """
        Инициализация на OAK-D камерата

        Args:
            rgb_resolution: "4K", "1080p", "720p" или "12MP"
            fps: кадри в секунда
            preview_size: размер на preview кадъра (width, height)
        """
        self.pipeline = None
        self.device = None
        self.q_rgb = None
        self.q_depth = None
        self.calibData = None

        # Настройки на резолюцията
        self.resolution_map = {
            "4K": dai.ColorCameraProperties.SensorResolution.THE_4_K,        # 3840x2160
            "12MP": dai.ColorCameraProperties.SensorResolution.THE_12_MP,    # 4056x3040
            "1080p": dai.ColorCameraProperties.SensorResolution.THE_1080_P,  # 1920x1080
            "720p": dai.ColorCameraProperties.SensorResolution.THE_720_P     # 1280x720
        }

        self.rgb_resolution = rgb_resolution
        self.fps = fps
        self.preview_size = preview_size
        self.is_opened = False

        # Настройване на пайплайна
        self._setup_pipeline()

    def _setup_pipeline(self):
        """Настройване на DepthAI пайплайна"""
        self.pipeline = dai.Pipeline()

        # RGB камера с максимална резолюция
        cam_rgb = self.pipeline.create(dai.node.ColorCamera)
        cam_rgb.setResolution(self.resolution_map[self.rgb_resolution])
        # Заменено RGB с CAM_A
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(
            dai.ColorCameraProperties.ColorOrder.BGR)  # OpenCV формат
        cam_rgb.setFps(self.fps)
        cam_rgb.initialControl.setManualFocus(120)
        cam_rgb.initialControl.setManualExposure(9000, 800)

        # Preview настройки
        if self.preview_size:
            cam_rgb.setPreviewSize(self.preview_size[0], self.preview_size[1])

        # Моно камери за depth
        mono_left = self.pipeline.create(dai.node.MonoCamera)
        mono_right = self.pipeline.create(dai.node.MonoCamera)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        mono_left.setResolution(
            dai.MonoCameraProperties.SensorResolution.THE_720_P)
        mono_right.setResolution(
            dai.MonoCameraProperties.SensorResolution.THE_720_P)
        mono_left.setFps(self.fps)
        mono_right.setFps(self.fps)

        # Stereo depth
        stereo = self.pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(
            dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        stereo.setLeftRightCheck(True)
        # Подравняване с CAM_A
        stereo.setOutputSize(2048, 1024)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

        # Връзки
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        # Изходи
        xout_rgb = self.pipeline.create(dai.node.XLinkOut)
        xout_depth = self.pipeline.create(dai.node.XLinkOut)
        xout_rgb_full = self.pipeline.create(dai.node.XLinkOut)

        xout_rgb.setStreamName("rgb")
        xout_depth.setStreamName("depth")
        xout_rgb_full.setStreamName("rgb_full")

        # Свързване на изходите
        cam_rgb.preview.link(xout_rgb.input)      # Preview винаги
        cam_rgb.video.link(xout_rgb_full.input)   # Пълна резолюция
        stereo.depth.link(xout_depth.input)

    def open(self):
        """Отваряне на камерата (еквивалент на cap.open())"""
        try:
            self.device = dai.Device(self.pipeline)

            # Настройване на опашките с по-големи буфери
            self.q_rgb = self.device.getOutputQueue(
                "rgb", maxSize=4, blocking=False)
            self.q_rgb_full = self.device.getOutputQueue(
                "rgb_full", maxSize=4, blocking=False)
            self.q_depth = self.device.getOutputQueue(
                "depth", maxSize=4, blocking=False)

            # Извличане на калибрационните данни
            self.calibData = self.device.readCalibration()

            # Изчакване за първи кадър
            import time
            time.sleep(1)  # Изчакване камерата да се стартира

            self.is_opened = True
            print(
                f"OAK-D камера отворена успешно с резолюция {self.rgb_resolution}")
            return True

        except Exception as e:
            print(f"Грешка при отваряне на OAK-D камерата: {e}")
            self.is_opened = False
            return False

    def isOpened(self):
        """Проверка дали камерата е отворена"""
        return self.is_opened

    def read(self, full_resolution=False, timeout_ms=1000):
        """
        Четене на кадър (еквивалент на cap.read())

        Args:
            full_resolution: False за preview, True за пълна резолюция
            timeout_ms: timeout в милисекунди за получаване на кадър

        Returns:
            ret: bool - успешно ли е прочетен кадъра
            frame: numpy array - кадъра
        """
        if not self.is_opened:
            return False, None

        try:
            if full_resolution:
                in_frame = self.q_rgb_full.get() if timeout_ms > 0 else self.q_rgb_full.tryGet()
            else:
                in_frame = self.q_rgb.get() if timeout_ms > 0 else self.q_rgb.tryGet()

            if in_frame is not None:
                frame = in_frame.getCvFrame()
                return True, frame
            else:
                return False, None

        except Exception as e:
            print(f"Грешка при четене на кадър: {e}")
            return False, None

    def get_depth_frame(self, timeout_ms=1000):
        """Получаване на depth кадър"""
        if not self.is_opened:
            return None

        try:
            in_depth = self.q_depth.get() if timeout_ms > 0 else self.q_depth.tryGet()
            if in_depth is not None:
                return in_depth.getFrame()
            return None
        except Exception as e:
            print(f"Грешка при четене на depth кадър: {e}")
            return None

    def get_both_frames(self, full_resolution=False):
        """Получаване на RGB и depth кадрите едновременно"""
        ret, rgb_frame = self.read(full_resolution)
        depth_frame = self.get_depth_frame()
        return ret, rgb_frame, depth_frame

    def get_intrinsics(self, full_resolution=False):
        """Получаване на калибрационните параметри"""
        if not self.calibData:
            return None

        if full_resolution:
            # За пълна резолюция
            if self.rgb_resolution == "4K":
                intrinsics = self.calibData.getCameraIntrinsics(
                    dai.CameraBoardSocket.CAM_A, 3840, 2160)
            elif self.rgb_resolution == "12MP":
                intrinsics = self.calibData.getCameraIntrinsics(
                    dai.CameraBoardSocket.CAM_A, 4056, 3040)
            elif self.rgb_resolution == "1080p":
                intrinsics = self.calibData.getCameraIntrinsics(
                    dai.CameraBoardSocket.CAM_A, 1920, 1080)
            elif self.rgb_resolution == "720p":
                intrinsics = self.calibData.getCameraIntrinsics(
                    dai.CameraBoardSocket.CAM_A, 1280, 720)
            else:
                intrinsics = self.calibData.getCameraIntrinsics(
                    dai.CameraBoardSocket.CAM_A)
        else:
            # За preview резолюция
            intrinsics = self.calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A,
                                                            self.preview_size[0], self.preview_size[1])

        fx, fy = intrinsics[0][0], intrinsics[1][1]
        cx, cy = intrinsics[0][2], intrinsics[1][2]

        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}

    def depth_to_xyz(self, depth_frame, mask=None, full_resolution=False):
        """
        Конвертира depth кадър в 3D точки

        Args:
            depth_frame: depth кадър от OAK-D
            mask: маска за филтриране на точки (optional)
            full_resolution: дали да използва калибрацията за пълна резолюция

        Returns:
            numpy array: 3D точки (N, 3) във формат [X, Y, Z]
        """
        intrinsics = self.get_intrinsics(full_resolution)
        if not intrinsics:
            return np.array([]).reshape(0, 3)

        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']

        if mask is not None:
            # Използване на маска
            ys, xs = np.where(mask)
        else:
            # Всички точки
            height, width = depth_frame.shape
            ys, xs = np.mgrid[0:height, 0:width]
            ys, xs = ys.flatten(), xs.flatten()

        # Проверка на границите
        height, width = depth_frame.shape
        valid_indices = (xs < width) & (ys < height) & (xs >= 0) & (ys >= 0)
        xs = xs[valid_indices]
        ys = ys[valid_indices]

        if len(xs) == 0:
            return np.array([]).reshape(0, 3)

        # Извличане на depth стойностите
        depths = depth_frame[ys, xs] / 1000.0  # mm → m

        # Конвертиране в 3D координати
        X = (xs - cx) * depths / fx
        Y = (ys - cy) * depths / fy
        Z = depths

        # Комбиниране в 3D точки
        pts3d = np.column_stack((X, Y, Z))

        # Премахване на точки с невалидна дълбочина
        valid_depth = Z > 0
        pts3d = pts3d[valid_depth]

        return pts3d

    def set(self, prop, value):
        """Настройване на свойства (ограничена функционалност)"""
        print(
            f"Настройка {prop} = {value} не се поддържа. Използвайте параметрите при инициализация.")
        return False

    def get(self, prop):
        """Получаване на свойства"""
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            if self.preview_size:
                return self.preview_size[0]
            elif self.rgb_resolution == "4K":
                return 3840
            elif self.rgb_resolution == "12MP":
                return 4056
            elif self.rgb_resolution == "1080p":
                return 1920
            else:
                return 1280
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            if self.preview_size:
                return self.preview_size[1]
            elif self.rgb_resolution == "4K":
                return 2160
            elif self.rgb_resolution == "12MP":
                return 3040
            elif self.rgb_resolution == "1080p":
                return 1080
            else:
                return 720
        elif prop == cv2.CAP_PROP_FPS:
            return self.fps
        else:
            return 0

    def release(self):
        """Затваряне на камерата (еквивалент на cap.release())"""
        if self.device:
            self.device.close()
            self.device = None
        self.is_opened = False
        print("OAK-D камера затворена")

    def __enter__(self):
        """Context manager вход"""
        if self.open():
            return self
        else:
            raise RuntimeError("Не може да се отвори OAK-D камерата")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager изход"""
        self.release()


# Помощни функции
def create_oakd_camera(resolution="4K", fps=30, preview_size=(1920, 1080)):
    """
    Фабрична функция за създаване на OAK-D камера

    Args:
        resolution: резолюция на камерата
        fps: кадри в секунда  
        preview_size: размер на preview

    Returns:
        OAKDCamera обект
    """
    return OAKDCamera(rgb_resolution=resolution, fps=fps, preview_size=preview_size)


def list_available_devices():
    """Списък на достъпните OAK-D устройства"""
    try:
        devices = dai.Device.getAllAvailableDevices()
        print(f"Намерени {len(devices)} OAK-D устройства:")
        for i, device in enumerate(devices):
            print(f"  {i}: {device.getMxId()} - {device.name}")
        return devices
    except Exception as e:
        print(f"Грешка при търсене на устройства: {e}")
        return []


if __name__ == "__main__":
    # Тест на модула
    print("OAK-D Camera Module Test")
    print("======================")

    # Списък на устройствата
    list_available_devices()

    # Тест на камерата
    with create_oakd_camera("4K", 30) as camera:
        print(f"Камера отворена: {camera.isOpened()}")

        intrinsics = camera.get_intrinsics()
        if intrinsics:
            print(
                f"Калибрация: fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}")

        # Тест четене на кадър
        ret, frame = camera.read()
        if ret:
            print(f"Кадър прочетен: {frame.shape}")
        else:
            print("Грешка при четене на кадър")

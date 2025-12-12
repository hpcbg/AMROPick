import cv2
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


class ObjectDetector:
    def find_markers(self, image, markers):
        markers = markers.values()
        marker_tags = [m['tag_id'] for m in markers]
        marker_coordinates = {m['tag_id']: m['global']
                              for m in markers}
        print("mc", marker_coordinates)
        dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_ARUCO_ORIGINAL)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        all_detected_markers, ids, rejected = detector.detectMarkers(gray)

        plane_markers = {}
        if ids is not None:
            ids = ids.flatten()
            pts = {}

            for corner, id_ in zip(all_detected_markers, ids):
                if str(id_) in marker_tags:
                    c = corner[0]
                    center = np.mean(c, axis=0)
                    pts[id_] = center
                    plane_markers[str(id_.item())] = {
                        "tag_id": str(id_.item()),
                        "local": {
                            "x": float(center[0]),
                            "y": float(center[1])
                        },
                        "global": marker_coordinates[str(id_)]
                    }

            if len(pts) != 4:
                plane_markers = None

        return plane_markers, all_detected_markers, ids

    def convert_to_global_coordinates(self, pick_pose, workarea):
        pose = workarea['pose']
        plane_markers = workarea['plane_markers'].values()
        locals = np.array([[m['local']['x'], m['local']['y']]
                          for m in plane_markers], dtype=np.float32)
        globals = np.array([[m['global']['x'], m['global']['y']]
                           for m in plane_markers], dtype=np.float32)
        H, _ = cv2.findHomography(locals, globals)
        pix = np.array([[pick_pose['x'], pick_pose['y'], 1]],
                       dtype=np.float32).T
        pt = H @ pix
        pt /= pt[2]
        coords = (pt[0][0], pt[1][0])

        x0, y0 = pose['x'], pose['y']
        theta0 = math.radians(pose['yaw'])
        x_l, y_l = float(coords[0]), float(coords[1])
        theta_l = math.radians(float(pick_pose['yaw']))

        x_g = x0 + x_l * math.cos(theta0) - y_l * math.sin(theta0)
        y_g = y0 + x_l * math.sin(theta0) + y_l * math.cos(theta0)
        theta_g = (theta0 + theta_l + math.pi) % (2 * math.pi) - math.pi
        theta_g = math.degrees(theta_g)

        return {"x": x_g, "y": y_g, "yaw": theta_g}

    def detect_object(self, image, masks, object_params=None, fence=None, plot=False):
        index = 0
        for mask in masks:
            image1 = image.copy()
            if plot:
                print("Mask index: ", index)
            index = index + 1

            binary = mask['segmentation'].astype(np.uint8)
            binary = cv2.resize(binary, (image1.shape[1], image1.shape[0]))

            try:
                kernel = np.ones((9, 9), np.uint8)
                cleaned1 = cv2.morphologyEx(
                    binary, cv2.MORPH_ERODE, kernel, iterations=5)
                cleaned1 = cv2.morphologyEx(
                    cleaned1, cv2.MORPH_DILATE, kernel, iterations=5)

                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    cleaned1, connectivity=8)

                largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

                cleaned2 = np.zeros_like(binary)
                cleaned2[labels == largest_component] = 255

                cleaned3 = cv2.medianBlur(cleaned2, 5)

                cleaned_combined = cv2.morphologyEx(
                    cleaned3, cv2.MORPH_CLOSE, kernel, iterations=1)
                binary = cleaned_combined.copy()
            except:
                continue
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            (cx, cy), (w1, h1), angle = rect
            w = max(w1, h1)
            h = min(w1, h1)
            area = cv2.contourArea(largest_contour)
            ratio = w/h
            area_ratio = area / (w * h)

            if fence is not None:
                if cx < fence['xmin'] or cx > fence['xmax'] or cy < fence['ymin'] or cy > fence['ymax']:
                    continue

                fence_pts = np.array([
                    [fence['xmin'], fence['ymin']],
                    [fence['xmin'], fence['ymax']],
                    [fence['xmax'], fence['ymax']],
                    [fence['xmax'], fence['ymin']]
                ], np.int32)
                cv2.polylines(image1, [fence_pts], True, (0, 0, 255), 2)

            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.drawContours(image1, [box], 0, (0, 255, 0), 2)

            cv2.circle(image1, (int(cx), int(cy)), radius=7, color=(
                0, 0, 255), thickness=-1)

            if plot:
                print("(w, h) = ", (w, h), "ratio = ", ratio, "area = ",
                      area, "area ratio = ", area_ratio)
                plt.figure(figsize=(40, 10))
                plt.subplot(121)
                plt.imshow(binary, 'gray')
                plt.subplot(122)
                plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
                plt.show()

            if object_params is not None:
                if area < object_params['area'] * ((100. - object_params['area_tolerance']) / 100.):
                    continue
                if area > object_params['area'] * ((100. + object_params['area_tolerance']) / 100.):
                    continue
                if ratio < object_params['ratio'] * ((100. - object_params['tolerance']) / 100.):
                    continue
                if ratio > object_params['ratio'] * ((100. + object_params['tolerance']) / 100.):
                    continue
                if area_ratio < object_params['area_ratio'] * ((100. - object_params['tolerance']) / 100.):
                    continue
                if area_ratio > object_params['area_ratio'] * ((100. + object_params['tolerance']) / 100.):
                    continue

                return image1, {'x': int(cx), 'y': int(cy), 'yaw': angle}

        return image, None

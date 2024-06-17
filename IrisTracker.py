import cv2 as cv
import mediapipe as mp
import numpy as np

class IrisTracker:
    def __init__(self, staticMode=False, maxfaces=1, minDetectionCon=0.8, minTrackCon=0.8, refine_landmarks=True):
        self.staticMode = staticMode
        self.maxfaces = maxfaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        self.refine_landmarks = refine_landmarks
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode,
                                                max_num_faces=self.maxfaces,
                                                min_detection_confidence=self.minDetectionCon,
                                                min_tracking_confidence=self.minTrackCon,
                                                refine_landmarks=self.refine_landmarks)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def GetFaceMesh(self, img, draw=True, fm_id=False, fm_thickness=0.5, fm_color=(255, 0, 0)):
        self.imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        self.faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                               self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])

                    if fm_id:
                        cv.putText(img, str(id), (x, y), cv.FONT_HERSHEY_COMPLEX_SMALL, fm_thickness, fm_color, 1)
                self.faces.append(face)
        return img

    def DrawIris(self, img, draw=True, iris='both', color=(0, 0, 255)):
        LeftIris = [474, 475, 476, 477]
        RightIris = [469, 470, 471, 472]

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        h, w = img.shape[:2]

        if self.results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [w, h]).astype(int) for p in self.results.multi_face_landmarks[0].landmark])

            if draw:
                if iris == 'left_eye' or iris == 0:
                    cv.polylines(img, [mesh_points[LeftIris]], True, color, 1, cv.LINE_AA)

                if iris == 'right_eye' or iris == 1:
                    cv.polylines(img, [mesh_points[RightIris]], True, color, 1, cv.LINE_AA)

                if iris == 'both' or iris == "" or iris == 2:
                    (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LeftIris])
                    (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RightIris])
                    center_left = np.array([l_cx, l_cy], dtype=np.int32)
                    center_right = np.array([r_cx, r_cy], dtype=np.int32)
                    cv.circle(img, center_left, int(l_radius), color, 1, cv.LINE_AA)
                    cv.circle(img, center_right, int(r_radius), color, 1, cv.LINE_AA)
            return center_left, center_right, img
        
        return None, None, img

def main():
    # Initialiser la capture vidéo
    cap = cv.VideoCapture(0)
    detector = IrisTracker(staticMode=False, maxfaces=1, minDetectionCon=0.5, minTrackCon=0.5)
    while True:
        _, img = cap.read()
        detector.GetFaceMesh(img, draw=False)
        left_center, right_center, img = detector.DrawIris(img, iris='both', draw=True)

        # Afficher l'image avec les annotations
        cv.imshow("Image", cv.flip(img, 1))

        # Quitter la boucle si 'q' est pressé
        key = cv.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
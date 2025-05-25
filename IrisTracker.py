import cv2 as cv
import mediapipe as mp
import numpy as np

class IrisTracker:
    def __init__(self, staticMode=False, maxfaces=1, minDetectionCon=0.8, minTrackCon=0.8, refine_landmarks=True):
        # Initialisation des paramètres pour le FaceMesh de MediaPipe
        self.staticMode = staticMode  # Mode image statique ou vidéo
        self.maxfaces = maxfaces      # Nombre max de visages à détecter
        self.minDetectionCon = minDetectionCon  # Seuil de confiance détection visage
        self.minTrackCon = minTrackCon            # Seuil de confiance suivi visage
        self.refine_landmarks = refine_landmarks  # Affiner les landmarks pour iris, lèvres, etc.

        # Utilitaires de dessin de MediaPipe
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh

        # Initialisation du modèle FaceMesh avec les paramètres spécifiés
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxfaces,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon,
            refine_landmarks=self.refine_landmarks
        )

        # Paramètres de dessin des landmarks (épaisseur et rayon des cercles)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def GetFaceMesh(self, img, draw=True, fm_id=False, fm_thickness=0.5, fm_color=(255, 0, 0)):
        # Convertir l'image en RGB car MediaPipe travaille en RGB
        self.imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # Traiter l'image pour détecter les landmarks du visage
        self.results = self.faceMesh.process(self.imgRGB)
        self.faces = []  # Liste pour stocker les coordonnées des points du visage

        # Si des visages sont détectés
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                # Dessiner les landmarks sur l'image si demandé
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                               self.drawSpec, self.drawSpec)

                face = []  # Stocke les coordonnées (x,y) des landmarks pour un visage
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)  # Convertir coordonnées normalisées en pixels
                    face.append([x, y])

                    # Afficher l'id du landmark sur l'image si demandé
                    if fm_id:
                        cv.putText(img, str(id), (x, y), cv.FONT_HERSHEY_COMPLEX_SMALL, fm_thickness, fm_color, 1)
                self.faces.append(face)  # Ajouter le visage à la liste
        return img

    def DrawIris(self, img, draw=True, iris='both', color=(0, 0, 255)):
        # Indices MediaPipe des points des iris gauche et droit
        LeftIris = [474, 475, 476, 477]
        RightIris = [469, 470, 471, 472]

        # Convertir en RGB et détecter les landmarks du visage (avec iris)
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        h, w = img.shape[:2]

        if self.results.multi_face_landmarks:
            # Récupérer tous les points du visage et les convertir en pixels
            mesh_points = np.array([np.multiply([p.x, p.y], [w, h]).astype(int) 
                                    for p in self.results.multi_face_landmarks[0].landmark])

            if draw:
                # Dessiner le contour de l'iris gauche si demandé
                if iris == 'left_eye' or iris == 0:
                    cv.polylines(img, [mesh_points[LeftIris]], True, color, 1, cv.LINE_AA)

                # Dessiner le contour de l'iris droit si demandé
                if iris == 'right_eye' or iris == 1:
                    cv.polylines(img, [mesh_points[RightIris]], True, color, 1, cv.LINE_AA)

                # Dessiner un cercle englobant autour des iris (les deux yeux)
                if iris == 'both' or iris == "" or iris == 2:
                    (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LeftIris])
                    (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RightIris])
                    center_left = np.array([l_cx, l_cy], dtype=np.int32)
                    center_right = np.array([r_cx, r_cy], dtype=np.int32)
                    cv.circle(img, center_left, int(l_radius), color, 1, cv.LINE_AA)
                    cv.circle(img, center_right, int(r_radius), color, 1, cv.LINE_AA)
            return center_left, center_right, img
        
        # Si aucun visage détecté, retourner None pour les centres
        return None, None, img

def main():
    # Initialiser la capture vidéo depuis la webcam
    cap = cv.VideoCapture(0)
    detector = IrisTracker(staticMode=False, maxfaces=1, minDetectionCon=0.5, minTrackCon=0.5)

    while True:
        # Lire une image depuis la webcam
        _, img = cap.read()
        # Extraire les landmarks du visage (sans dessiner)
        detector.GetFaceMesh(img, draw=False)
        # Dessiner les iris détectés et récupérer leurs centres
        left_center, right_center, img = detector.DrawIris(img, iris='both', draw=True)

        # Afficher l'image retournée horizontalement (effet miroir)
        cv.imshow("Image", cv.flip(img, 1))

        # Quitter si la touche 'q' est pressée
        key = cv.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    # Libérer la caméra et fermer les fenêtres
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()

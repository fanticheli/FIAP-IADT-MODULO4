"""
Analisador de atividades usando MediaPipe Pose Landmarker (Tasks API).
"""
import math
from collections import deque
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.analyzers.base_analyzer import BaseAnalyzer
from src.models.detection import ActivityDetection
from src.utils.logger import logger


# Traducao das atividades para portugues
ACTIVITY_LABELS = {
    "standing": "Em pe",
    "sitting": "Sentado",
    "arms_raised": "Bracos levantados",
    "leaning": "Inclinado",
}

# Indices dos landmarks de pose
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# Visibilidade minima para considerar um landmark confiavel
MIN_VISIBILITY = 0.5

# Tamanho do historico para suavizacao temporal
SMOOTHING_WINDOW = 5

# Limiares para deteccao de anomalias
VELOCITY_THRESHOLD = 0.15  # Movimento brusco se velocidade > 15% do frame
EXTREME_ANGLE_MIN = 30  # Angulo muito fechado (graus)
EXTREME_ANGLE_MAX = 160  # Angulo muito aberto (graus)
TRUNK_LEAN_THRESHOLD = 45  # Inclinacao extrema do tronco (graus)


class ActivityAnalyzer(BaseAnalyzer):
    """
    Analisador de atividades usando MediaPipe Pose Landmarker.

    Detecta atividades basicas: em pe, sentado, bracos levantados, inclinado.
    """

    # URL do modelo - usando FULL para maior precisao
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
    MODEL_NAME = "pose_landmarker_full.task"

    def __init__(self, min_detection_confidence: float = 0.5):
        self._min_confidence = min_detection_confidence
        self._landmarker = None
        self._activity_counter = 0
        self._activity_history: deque = deque(maxlen=SMOOTHING_WINDOW)
        self._prev_landmarks = None  # Landmarks do frame anterior para calcular velocidade

    @property
    def name(self) -> str:
        return "activity_detection"

    def _get_model_path(self) -> str:
        """Obtém o caminho do modelo ou baixa se necessário."""
        import urllib.request
        from pathlib import Path

        # Diretório para armazenar o modelo
        model_dir = Path(__file__).parent.parent.parent / "models"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / self.MODEL_NAME

        if not model_path.exists():
            logger.info("Baixando modelo de pose...")
            urllib.request.urlretrieve(self.MODEL_URL, model_path)
            logger.info(f"Modelo salvo em: {model_path}")

        return str(model_path)

    def setup(self) -> None:
        """Inicializa o detector de pose."""
        self._activity_counter = 0
        self._activity_history.clear()
        self._prev_landmarks = None

        # Configura o landmarker
        base_options = python.BaseOptions(
            model_asset_path=self._get_model_path()
        )
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            min_pose_detection_confidence=self._min_confidence,
            min_tracking_confidence=0.5
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

        logger.info("ActivityAnalyzer inicializado (MediaPipe Pose Full)")

    def _calculate_angle(self, p1, p2, p3) -> float:
        """
        Calcula o angulo entre tres pontos (em graus).
        O angulo e formado em p2.
        """
        # Vetores
        v1 = (p1.x - p2.x, p1.y - p2.y)
        v2 = (p3.x - p2.x, p3.y - p2.y)

        # Produto escalar e magnitudes
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

        if mag1 * mag2 == 0:
            return 0.0

        # Angulo em graus
        cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_angle))

    def _is_visible(self, *landmarks) -> bool:
        """Verifica se todos os landmarks tem visibilidade suficiente."""
        return all(lm.visibility >= MIN_VISIBILITY for lm in landmarks)

    def _get_smoothed_activity(self, current_activity: str) -> str:
        """
        Aplica suavizacao temporal usando votacao por maioria.
        Evita 'flickering' entre classificacoes.
        """
        self._activity_history.append(current_activity)

        if len(self._activity_history) < 3:
            return current_activity

        # Conta ocorrencias de cada atividade no historico
        counts = {}
        for act in self._activity_history:
            counts[act] = counts.get(act, 0) + 1

        # Retorna a atividade mais frequente
        return max(counts, key=counts.get)

    def _calculate_landmarks_velocity(self, landmarks) -> float:
        """
        Calcula a velocidade media dos landmarks entre o frame atual e o anterior.
        Retorna um valor normalizado (0.0 a 1.0+).
        """
        if self._prev_landmarks is None:
            return 0.0

        total_displacement = 0.0
        valid_points = 0

        # Pontos principais para medir movimento
        key_points = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
                      LEFT_WRIST, RIGHT_WRIST, LEFT_KNEE, RIGHT_KNEE]

        for idx in key_points:
            curr = landmarks[idx]
            prev = self._prev_landmarks[idx]

            # Só considera pontos com boa visibilidade em ambos os frames
            if curr.visibility >= MIN_VISIBILITY and prev.visibility >= MIN_VISIBILITY:
                dx = curr.x - prev.x
                dy = curr.y - prev.y
                displacement = math.sqrt(dx**2 + dy**2)
                total_displacement += displacement
                valid_points += 1

        if valid_points == 0:
            return 0.0

        return total_displacement / valid_points

    def _calculate_body_angles(self, landmarks) -> dict:
        """
        Calcula angulos corporais importantes para detectar poses atipicas.
        """
        angles = {}

        # Angulo do tronco (inclinacao lateral/frontal)
        left_shoulder = landmarks[LEFT_SHOULDER]
        right_shoulder = landmarks[RIGHT_SHOULDER]
        left_hip = landmarks[LEFT_HIP]
        right_hip = landmarks[RIGHT_HIP]

        if self._is_visible(left_shoulder, right_shoulder, left_hip, right_hip):
            # Centro dos ombros e quadril
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_center_x = (left_hip.x + right_hip.x) / 2
            hip_center_y = (left_hip.y + right_hip.y) / 2

            # Angulo do tronco em relacao a vertical
            dx = shoulder_center_x - hip_center_x
            dy = shoulder_center_y - hip_center_y

            if abs(dy) > 0.01:
                trunk_angle = abs(math.degrees(math.atan(dx / dy)))
                angles["trunk"] = trunk_angle

        # Angulo do cotovelo esquerdo
        left_elbow = landmarks[LEFT_ELBOW]
        left_wrist = landmarks[LEFT_WRIST]
        if self._is_visible(left_shoulder, left_elbow, left_wrist):
            angles["left_elbow"] = self._calculate_angle(left_shoulder, left_elbow, left_wrist)

        # Angulo do cotovelo direito
        right_elbow = landmarks[RIGHT_ELBOW]
        right_wrist = landmarks[RIGHT_WRIST]
        if self._is_visible(right_shoulder, right_elbow, right_wrist):
            angles["right_elbow"] = self._calculate_angle(right_shoulder, right_elbow, right_wrist)

        # Angulo do joelho esquerdo
        left_knee = landmarks[LEFT_KNEE]
        left_ankle = landmarks[LEFT_ANKLE]
        if self._is_visible(left_hip, left_knee, left_ankle):
            angles["left_knee"] = self._calculate_angle(left_hip, left_knee, left_ankle)

        # Angulo do joelho direito
        right_knee = landmarks[RIGHT_KNEE]
        right_ankle = landmarks[RIGHT_ANKLE]
        if self._is_visible(right_hip, right_knee, right_ankle):
            angles["right_knee"] = self._calculate_angle(right_hip, right_knee, right_ankle)

        return angles

    def _detect_anomaly(self, velocity: float, angles: dict) -> tuple[bool, str]:
        """
        Detecta se ha anomalia no movimento ou pose.

        Returns:
            Tupla (is_anomaly, anomaly_type)
        """
        anomalies = []

        # 1. Movimento brusco (velocidade alta)
        if velocity > VELOCITY_THRESHOLD:
            anomalies.append(f"Movimento brusco (vel={velocity:.2f})")

        # 2. Pose atipica - tronco muito inclinado
        if "trunk" in angles and angles["trunk"] > TRUNK_LEAN_THRESHOLD:
            anomalies.append(f"Tronco inclinado ({angles['trunk']:.0f}°)")

        # 3. Pose atipica - angulos extremos nos membros
        for joint in ["left_elbow", "right_elbow", "left_knee", "right_knee"]:
            if joint in angles:
                angle = angles[joint]
                if angle < EXTREME_ANGLE_MIN:
                    joint_name = joint.replace("_", " ").title()
                    anomalies.append(f"{joint_name} muito dobrado ({angle:.0f}°)")
                elif angle > EXTREME_ANGLE_MAX:
                    joint_name = joint.replace("_", " ").title()
                    anomalies.append(f"{joint_name} hiperestendido ({angle:.0f}°)")

        if anomalies:
            return True, "; ".join(anomalies)

        return False, ""

    def _classify_activity(self, landmarks) -> tuple[str, float]:
        """
        Classifica a atividade baseado nos landmarks da pose.
        Usa angulos e proporcoes para maior precisao.

        Returns:
            Tupla (activity_key, confidence)
        """
        # Pontos importantes
        nose = landmarks[NOSE]
        left_shoulder = landmarks[LEFT_SHOULDER]
        right_shoulder = landmarks[RIGHT_SHOULDER]
        left_elbow = landmarks[LEFT_ELBOW]
        right_elbow = landmarks[RIGHT_ELBOW]
        left_wrist = landmarks[LEFT_WRIST]
        right_wrist = landmarks[RIGHT_WRIST]
        left_hip = landmarks[LEFT_HIP]
        right_hip = landmarks[RIGHT_HIP]
        left_knee = landmarks[LEFT_KNEE]
        right_knee = landmarks[RIGHT_KNEE]
        left_ankle = landmarks[LEFT_ANKLE]
        right_ankle = landmarks[RIGHT_ANKLE]

        # Calcula visibilidade media dos landmarks principais
        core_landmarks = [left_shoulder, right_shoulder, left_hip, right_hip]
        visibility = sum(lm.visibility for lm in core_landmarks) / len(core_landmarks)

        # Se visibilidade muito baixa, retorna standing como fallback
        if visibility < MIN_VISIBILITY:
            return "standing", visibility

        # === DETECCAO DE BRACOS LEVANTADOS ===
        # Verifica se os pulsos estao acima dos ombros E cotovelos
        arms_raised = False
        if self._is_visible(left_wrist, left_shoulder, left_elbow):
            left_arm_up = left_wrist.y < left_shoulder.y and left_wrist.y < left_elbow.y
        else:
            left_arm_up = False

        if self._is_visible(right_wrist, right_shoulder, right_elbow):
            right_arm_up = right_wrist.y < right_shoulder.y and right_wrist.y < right_elbow.y
        else:
            right_arm_up = False

        # Pelo menos um braco levantado significativamente
        if left_arm_up or right_arm_up:
            # Calcula angulo do braco em relacao ao corpo
            if left_arm_up and self._is_visible(left_wrist, left_shoulder, left_hip):
                angle = self._calculate_angle(left_wrist, left_shoulder, left_hip)
                if angle > 120:  # Braco bem levantado
                    arms_raised = True

            if right_arm_up and self._is_visible(right_wrist, right_shoulder, right_hip):
                angle = self._calculate_angle(right_wrist, right_shoulder, right_hip)
                if angle > 120:  # Braco bem levantado
                    arms_raised = True

        if arms_raised:
            return "arms_raised", visibility

        # === DETECCAO DE SENTADO ===
        # Usa angulo do joelho: sentado tem joelho mais dobrado (angulo menor)
        sitting = False

        if self._is_visible(left_hip, left_knee, left_ankle):
            left_knee_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
            left_sitting = left_knee_angle < 120  # Joelho dobrado
        else:
            left_sitting = False

        if self._is_visible(right_hip, right_knee, right_ankle):
            right_knee_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
            right_sitting = right_knee_angle < 120  # Joelho dobrado
        else:
            right_sitting = False

        # Tambem verifica a relacao de altura quadril-joelho
        hip_y = (left_hip.y + right_hip.y) / 2
        knee_y = (left_knee.y + right_knee.y) / 2
        hip_knee_ratio = abs(knee_y - hip_y)

        # Sentado: joelhos dobrados OU joelhos na mesma altura do quadril
        if (left_sitting and right_sitting) or hip_knee_ratio < 0.08:
            sitting = True

        if sitting:
            return "sitting", visibility

        # === DETECCAO DE INCLINADO ===
        # Calcula inclinacao do tronco usando ombros e quadril
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
        hip_x = (left_hip.x + right_hip.x) / 2

        # Angulo do tronco em relacao a vertical
        trunk_height = hip_y - shoulder_y
        trunk_horizontal = abs(hip_x - shoulder_x)

        # Se a diferenca horizontal for significativa em relacao a altura
        if trunk_height > 0.05:  # Evita divisao por valores muito pequenos
            lean_ratio = trunk_horizontal / trunk_height
            if lean_ratio > 0.3:  # Inclinacao significativa
                return "leaning", visibility

        # Tambem detecta inclinacao quando o nariz esta muito a frente
        if self._is_visible(nose):
            nose_hip_horizontal = abs(nose.x - hip_x)
            if nose_hip_horizontal > 0.15:
                return "leaning", visibility

        # === DEFAULT: EM PE ===
        return "standing", visibility

    def analyze(
        self,
        frame: np.ndarray,
        frame_number: int
    ) -> list[ActivityDetection]:
        """
        Analisa atividades no frame.

        Args:
            frame: Frame em formato BGR (OpenCV).
            frame_number: Numero do frame.

        Returns:
            Lista de ActivityDetection encontradas.
        """
        detections = []

        if self._landmarker is None:
            self.setup()

        # Converte BGR para RGB
        rgb_frame = frame[:, :, ::-1].copy()

        # Cria imagem MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Processa o frame
        results = self._landmarker.detect(mp_image)

        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            self._activity_counter += 1

            # Pega os landmarks da primeira pose detectada
            landmarks = results.pose_landmarks[0]

            # Classifica a atividade
            raw_activity, confidence = self._classify_activity(landmarks)

            # Aplica suavizacao temporal para evitar flickering
            activity = self._get_smoothed_activity(raw_activity)

            # Calcula velocidade e angulos para deteccao de anomalias
            velocity = self._calculate_landmarks_velocity(landmarks)
            angles = self._calculate_body_angles(landmarks)

            # Detecta anomalias
            is_anomaly, anomaly_type = self._detect_anomaly(velocity, angles)

            detection = ActivityDetection(
                activity_id=self._activity_counter,
                activity=activity,
                activity_label=ACTIVITY_LABELS.get(activity, activity),
                confidence=confidence,
                frame_number=frame_number,
                landmarks_velocity=velocity,
                pose_angles=angles,
                is_anomaly=is_anomaly,
                anomaly_type=anomaly_type
            )
            detections.append(detection)

            # Armazena landmarks para o proximo frame
            self._prev_landmarks = landmarks

        return detections

    def teardown(self) -> None:
        """Libera recursos."""
        self._landmarker = None
        self._activity_history.clear()
        self._prev_landmarks = None
        logger.info(f"ActivityAnalyzer finalizado. Total: {self._activity_counter} deteccoes")

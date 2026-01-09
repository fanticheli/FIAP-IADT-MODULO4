"""
Analisador de expressoes emocionais usando DeepFace.
"""
import numpy as np
from deepface import DeepFace

from src.analyzers.base_analyzer import BaseAnalyzer
from src.models.detection import EmotionDetection
from src.utils.logger import logger


# Traducao das emocoes para portugues
EMOTION_LABELS = {
    "angry": "Raiva",
    "disgust": "Nojo",
    "fear": "Medo",
    "happy": "Feliz",
    "sad": "Triste",
    "surprise": "Surpresa",
    "neutral": "Neutro"
}

# Tamanho minimo do rosto em pixels (largura e altura)
MIN_FACE_SIZE = 30

# Confianca minima do rosto detectado
MIN_FACE_CONFIDENCE = 0.5


class EmotionAnalyzer(BaseAnalyzer):
    """
    Analisador de expressoes emocionais usando DeepFace.

    Detecta 7 emocoes: raiva, nojo, medo, feliz, triste, surpresa, neutro.
    """

    def __init__(self):
        self._emotion_counter = 0
        self._emotion_counts = {emotion: 0 for emotion in EMOTION_LABELS.keys()}

    @property
    def name(self) -> str:
        return "emotion_detection"

    def setup(self) -> None:
        """Inicializa o analisador."""
        self._emotion_counter = 0
        self._emotion_counts = {emotion: 0 for emotion in EMOTION_LABELS.keys()}
        logger.info("EmotionAnalyzer inicializado (DeepFace)")

    def analyze(
        self,
        frame: np.ndarray,
        frame_number: int
    ) -> list[EmotionDetection]:
        """
        Analisa emocoes nos rostos do frame.

        Args:
            frame: Frame em formato BGR (OpenCV).
            frame_number: Numero do frame.

        Returns:
            Lista de EmotionDetection encontradas.
        """
        detections = []

        try:
            # DeepFace analisa emocoes
            # Usa opencv como detector (mais rapido e compativel)
            results = DeepFace.analyze(
                frame,
                actions=['emotion'],
                detector_backend='opencv',
                enforce_detection=False,
                silent=True
            )

            # Garante que results seja lista
            if not isinstance(results, list):
                results = [results]

            for result in results:
                emotions = result.get("emotion", {})
                region = result.get("region", {})

                # Bounding box
                x = region.get("x", 0)
                y = region.get("y", 0)
                w = region.get("w", 0)
                h = region.get("h", 0)

                # Valida se o rosto detectado e real
                # 1. Bounding box deve ter tamanho minimo
                if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                    continue

                # 2. Bounding box deve estar dentro do frame
                frame_h, frame_w = frame.shape[:2]
                if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
                    continue

                # 3. Verifica confianca do rosto (se disponivel)
                face_confidence = result.get("face_confidence", 1.0)
                if face_confidence < MIN_FACE_CONFIDENCE:
                    continue

                if emotions:
                    self._emotion_counter += 1

                    # Pega a emocao dominante
                    dominant = result.get("dominant_emotion", "neutral")
                    confidence = emotions.get(dominant, 0) / 100

                    # Conta emocoes
                    self._emotion_counts[dominant] = self._emotion_counts.get(dominant, 0) + 1

                    detection = EmotionDetection(
                        emotion_id=self._emotion_counter,
                        emotion=dominant,
                        emotion_label=EMOTION_LABELS.get(dominant, dominant),
                        confidence=confidence,
                        frame_number=frame_number,
                        box=(x, y, w, h),
                        all_emotions=emotions
                    )
                    detections.append(detection)

        except Exception:
            # Se nao encontrar rosto ou erro, retorna vazio
            pass

        return detections

    def teardown(self) -> None:
        """Libera recursos."""
        logger.info(f"EmotionAnalyzer finalizado. Total: {self._emotion_counter} deteccoes")

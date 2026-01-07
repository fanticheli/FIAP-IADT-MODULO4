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
                self._emotion_counter += 1

                emotions = result.get("emotion", {})
                region = result.get("region", {})

                if emotions:
                    # Pega a emocao dominante
                    dominant = result.get("dominant_emotion", "neutral")
                    confidence = emotions.get(dominant, 0) / 100

                    # Conta emocoes
                    self._emotion_counts[dominant] = self._emotion_counts.get(dominant, 0) + 1

                    # Bounding box
                    x = region.get("x", 0)
                    y = region.get("y", 0)
                    w = region.get("w", 0)
                    h = region.get("h", 0)

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

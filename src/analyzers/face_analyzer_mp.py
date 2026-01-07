"""
Analisador de detecção facial usando MediaPipe (nova API tasks).
Mais rápido e preciso que face_recognition em CPU.
"""
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.analyzers.base_analyzer import BaseAnalyzer
from src.models.detection import BoundingBox, FaceDetection
from src.utils.logger import logger


class FaceAnalyzerMediaPipe(BaseAnalyzer):
    """
    Analisador para detecção de rostos usando MediaPipe Tasks API.

    Vantagens sobre face_recognition:
    - Muito mais rápido em CPU
    - Detecta rostos em vários ângulos
    - Mais robusto a condições de iluminação
    """

    # URL do modelo
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    MODEL_NAME = "blaze_face_short_range.tflite"

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
    ):
        """
        Inicializa o analisador facial MediaPipe.

        Args:
            min_detection_confidence: Confiança mínima para detecção (0.0-1.0).
                                     Use valores baixos (0.3-0.4) para rostos distantes.
        """
        self._min_confidence = min_detection_confidence
        self._detector = None
        self._face_counter = 0

    @property
    def name(self) -> str:
        return "face_detection"

    def setup(self) -> None:
        """Inicializa o detector MediaPipe."""
        self._face_counter = 0

        # Configura o detector
        base_options = python.BaseOptions(
            model_asset_path=self._get_model_path()
        )
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=self._min_confidence
        )
        self._detector = vision.FaceDetector.create_from_options(options)

        logger.info(f"FaceAnalyzerMediaPipe inicializado (confidence={self._min_confidence})")
        logger.info(f"  Dica: Use --confidence 0.3 para detectar rostos mais distantes")

    def _get_model_path(self) -> str:
        """Obtém o caminho do modelo ou baixa se necessário."""
        import urllib.request
        from pathlib import Path

        # Diretório para armazenar o modelo
        model_dir = Path(__file__).parent.parent.parent / "models"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / self.MODEL_NAME

        if not model_path.exists():
            logger.info(f"Baixando modelo de detecção facial...")
            urllib.request.urlretrieve(self.MODEL_URL, model_path)
            logger.info(f"Modelo salvo em: {model_path}")

        return str(model_path)

    def analyze(
        self,
        frame: np.ndarray,
        frame_number: int
    ) -> list[FaceDetection]:
        """
        Detecta rostos no frame usando MediaPipe.

        Args:
            frame: Frame em formato BGR (OpenCV).
            frame_number: Número do frame.

        Returns:
            Lista de FaceDetection encontradas.
        """
        if self._detector is None:
            self.setup()

        # Converte BGR para RGB
        rgb_frame = frame[:, :, ::-1].copy()

        # Cria imagem MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Processa o frame
        results = self._detector.detect(mp_image)

        detections = []
        height, width = frame.shape[:2]

        for detection in results.detections:
            self._face_counter += 1

            # Obtém bounding box
            bbox = detection.bounding_box

            # Coordenadas já são absolutas na nova API
            left = max(0, bbox.origin_x)
            top = max(0, bbox.origin_y)
            right = min(width, bbox.origin_x + bbox.width)
            bottom = min(height, bbox.origin_y + bbox.height)

            # Obtém confiança
            confidence = None
            if detection.categories:
                confidence = detection.categories[0].score

            face_det = FaceDetection(
                face_id=self._face_counter,
                bounding_box=BoundingBox(
                    top=top,
                    right=right,
                    bottom=bottom,
                    left=left
                ),
                frame_number=frame_number,
                confidence=confidence
            )
            detections.append(face_det)

        return detections

    def teardown(self) -> None:
        """Libera recursos do MediaPipe."""
        self._detector = None
        logger.info(f"FaceAnalyzerMediaPipe finalizado. Total de detecções: {self._face_counter}")

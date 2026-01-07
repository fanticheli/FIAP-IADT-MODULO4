"""
Configurações globais da aplicação de análise de vídeo.
"""
from pathlib import Path


class Settings:
    """Configurações centralizadas do projeto."""

    # Diretórios base
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    VIDEOS_DIR = BASE_DIR / "videos"
    OUTPUT_DIR = BASE_DIR / "resultado"

    # Configurações de processamento de vídeo
    FRAME_SAMPLE_RATE = 1  # Processar 1 a cada N frames (1 = todos)

    # Configurações de detecção facial
    FACE_DETECTOR = "mediapipe"  # "mediapipe" (recomendado), "hog" ou "cnn"
    FACE_DETECTION_MODEL = "hog"  # Para face_recognition: "hog" (CPU) ou "cnn" (GPU)
    FACE_DETECTION_UPSAMPLE = 1  # Número de vezes para aumentar a imagem
    MEDIAPIPE_CONFIDENCE = 0.5  # Confiança mínima para MediaPipe (0.0-1.0)

    # Configurações de visualização
    FACE_BOX_COLOR = (0, 255, 0)  # Verde (BGR)
    FACE_BOX_THICKNESS = 2
    FACE_LABEL_FONT_SCALE = 0.6
    FACE_LABEL_COLOR = (255, 255, 255)  # Branco (BGR)

    # Configurações de saída
    OUTPUT_VIDEO_CODEC = "mp4v"
    OUTPUT_VIDEO_FPS = None  # None = usar FPS original do vídeo

    @classmethod
    def ensure_output_dir(cls) -> Path:
        """Garante que o diretório de saída existe."""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return cls.OUTPUT_DIR


# Instância singleton para uso global
settings = Settings()

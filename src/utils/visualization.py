"""
Utilitários para visualização e anotação de frames.
"""
import cv2
import numpy as np

from src.config.settings import settings
from src.models.detection import FaceDetection, EmotionDetection, ActivityDetection


def draw_face_detection(
    frame: np.ndarray,
    detection: FaceDetection,
    color: tuple[int, int, int] = None,
    thickness: int = None,
    show_label: bool = True
) -> np.ndarray:
    """
    Desenha uma detecção facial no frame.

    Args:
        frame: Frame onde desenhar (será modificado in-place).
        detection: Detecção facial a ser desenhada.
        color: Cor do retângulo em BGR.
        thickness: Espessura da linha.
        show_label: Se deve mostrar o ID do rosto.

    Returns:
        Frame com a anotação desenhada.
    """
    color = color or settings.FACE_BOX_COLOR
    thickness = thickness or settings.FACE_BOX_THICKNESS

    bbox = detection.bounding_box
    top_left = (bbox.left, bbox.top)
    bottom_right = (bbox.right, bbox.bottom)

    # Desenha o retângulo
    cv2.rectangle(frame, top_left, bottom_right, color, thickness)

    # Desenha o label com o ID
    if show_label:
        label = f"Face #{detection.face_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = settings.FACE_LABEL_FONT_SCALE
        label_color = settings.FACE_LABEL_COLOR

        # Calcula tamanho do texto para o background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, 1
        )

        # Posição do texto (acima do retângulo)
        text_x = bbox.left
        text_y = bbox.top - 10 if bbox.top > 30 else bbox.bottom + 20

        # Background do texto
        cv2.rectangle(
            frame,
            (text_x, text_y - text_height - 5),
            (text_x + text_width + 5, text_y + 5),
            color,
            -1  # Preenchido
        )

        # Texto
        cv2.putText(
            frame,
            label,
            (text_x + 2, text_y),
            font,
            font_scale,
            label_color,
            1,
            cv2.LINE_AA
        )

    return frame


def draw_all_detections(
    frame: np.ndarray,
    detections: list[FaceDetection]
) -> np.ndarray:
    """
    Desenha todas as detecções faciais no frame.

    Args:
        frame: Frame onde desenhar.
        detections: Lista de detecções faciais.

    Returns:
        Frame com todas as anotações desenhadas.
    """
    for detection in detections:
        draw_face_detection(frame, detection)
    return frame


def draw_frame_info(
    frame: np.ndarray,
    frame_number: int,
    face_count: int,
    fps: float = None
) -> np.ndarray:
    """
    Desenha informações do frame no canto superior esquerdo.

    Args:
        frame: Frame onde desenhar.
        frame_number: Número do frame atual.
        face_count: Quantidade de rostos detectados.
        fps: FPS do vídeo (opcional).

    Returns:
        Frame com as informações desenhadas.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (0, 255, 255)  # Amarelo
    thickness = 2

    info_text = f"Frame: {frame_number} | Faces: {face_count}"
    if fps:
        time_sec = frame_number / fps
        info_text += f" | Time: {time_sec:.1f}s"

    cv2.putText(
        frame,
        info_text,
        (10, 30),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )

    return frame


def draw_emotion_detection(
    frame: np.ndarray,
    detection: EmotionDetection
) -> np.ndarray:
    """
    Desenha uma detecção de emoção no frame.

    Args:
        frame: Frame onde desenhar.
        detection: Detecção de emoção.

    Returns:
        Frame com a anotação desenhada.
    """
    x, y, w, h = detection.box
    color = (255, 165, 0)  # Laranja (BGR)

    # Desenha retângulo
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Label com emoção
    label = f"{detection.emotion_label} ({detection.confidence:.0%})"
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Background do texto
    (text_w, text_h), _ = cv2.getTextSize(label, font, 0.5, 1)
    cv2.rectangle(frame, (x, y + h), (x + text_w + 5, y + h + text_h + 10), color, -1)

    # Texto
    cv2.putText(frame, label, (x + 2, y + h + text_h + 5), font, 0.5, (255, 255, 255), 1)

    return frame


def draw_all_emotions(
    frame: np.ndarray,
    emotions: list[EmotionDetection]
) -> np.ndarray:
    """
    Desenha todas as detecções de emoção no frame.
    """
    for emotion in emotions:
        draw_emotion_detection(frame, emotion)
    return frame


def draw_activity_detection(
    frame: np.ndarray,
    detection: ActivityDetection
) -> np.ndarray:
    """
    Desenha uma detecção de atividade no frame.

    Args:
        frame: Frame onde desenhar.
        detection: Detecção de atividade.

    Returns:
        Frame com a anotação desenhada.
    """
    height, width = frame.shape[:2]
    color = (0, 200, 200)  # Amarelo-verde (BGR)

    # Label com atividade (canto inferior direito)
    label = f"Atividade: {detection.activity_label}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7

    # Background do texto
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, 2)
    x = width - text_w - 15
    y = height - 20

    cv2.rectangle(frame, (x - 5, y - text_h - 5), (x + text_w + 5, y + 5), color, -1)
    cv2.putText(frame, label, (x, y), font, font_scale, (0, 0, 0), 2)

    return frame


def draw_all_activities(
    frame: np.ndarray,
    activities: list[ActivityDetection]
) -> np.ndarray:
    """
    Desenha todas as detecções de atividade no frame.
    """
    for activity in activities:
        draw_activity_detection(frame, activity)
    return frame

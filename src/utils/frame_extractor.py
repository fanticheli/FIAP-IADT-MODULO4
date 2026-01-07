"""
Extrator de frames de video.
"""
import random
import cv2
from pathlib import Path

from src.utils.logger import logger


def extrair_frames_anomalias(
    video_path: Path,
    anomalias: list[dict],
    output_dir: Path,
    max_frames: int = 3
) -> list[Path]:
    """
    Extrai frames aleatorios onde ocorreram anomalias.

    Args:
        video_path: Caminho do video.
        anomalias: Lista de anomalias com numero do frame.
        output_dir: Diretorio para salvar as imagens.
        max_frames: Maximo de frames a extrair.

    Returns:
        Lista de caminhos das imagens salvas.
    """
    if not anomalias:
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Nao foi possivel abrir o video: {video_path}")
        return []

    # Seleciona anomalias aleatorias
    anomalias_selecionadas = random.sample(
        anomalias,
        min(max_frames, len(anomalias))
    )

    imagens_salvas = []

    for anomalia in anomalias_selecionadas:
        frame_num = anomalia.get('frame', 0)
        tipo = anomalia.get('tipo', 'Anomalia')
        descricao = anomalia.get('descricao', '')

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if ret:
            # Adiciona indicacao visual da anomalia
            frame = _adicionar_indicacao(frame, frame_num, tipo, descricao)

            # Salva o frame
            img_path = output_dir / f"anomalia_frame_{frame_num}.jpg"
            cv2.imwrite(str(img_path), frame)
            imagens_salvas.append(img_path)
            logger.info(f"Frame de anomalia salvo: {img_path}")

    cap.release()
    return imagens_salvas


def _adicionar_indicacao(
    frame,
    frame_num: int,
    tipo: str,
    descricao: str
) -> any:
    """
    Adiciona indicacao visual da anomalia no frame.
    """
    height, width = frame.shape[:2]

    # Fundo semi-transparente no topo
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    # Texto indicando a anomalia
    cv2.putText(
        frame,
        f"ANOMALIA DETECTADA - Frame {frame_num}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),  # Vermelho
        2
    )

    cv2.putText(
        frame,
        f"{tipo}: {descricao}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),  # Branco
        2
    )

    # Borda vermelha indicando anomalia
    cv2.rectangle(frame, (5, 5), (width - 5, height - 5), (0, 0, 255), 3)

    return frame

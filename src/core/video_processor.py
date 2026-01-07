"""
Processador principal de vídeo.
Orquestra a análise de vídeo utilizando diferentes analisadores.
"""
from pathlib import Path
from typing import Optional

import cv2
from tqdm import tqdm

from src.analyzers.base_analyzer import BaseAnalyzer
from src.config.settings import settings
from src.models.detection import FrameAnalysis, VideoAnalysisResult
from src.utils.logger import logger
from src.utils.visualization import draw_all_detections, draw_all_emotions, draw_all_activities, draw_frame_info


class VideoProcessor:
    """
    Processador de vídeo que aplica múltiplos analisadores.

    Utiliza o padrão Strategy para permitir diferentes tipos de análise
    de forma modular e extensível.
    """

    def __init__(
        self,
        analyzers: list[BaseAnalyzer],
        frame_sample_rate: Optional[int] = None
    ):
        """
        Inicializa o processador de vídeo.

        Args:
            analyzers: Lista de analisadores a serem aplicados.
            frame_sample_rate: Processa 1 a cada N frames (1 = todos).
        """
        self._analyzers = analyzers
        self._frame_sample_rate = frame_sample_rate or settings.FRAME_SAMPLE_RATE

    def process(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        show_preview: bool = False
    ) -> VideoAnalysisResult:
        """
        Processa um vídeo aplicando todos os analisadores.

        Args:
            video_path: Caminho do vídeo de entrada.
            output_path: Caminho do vídeo de saída (opcional).
            show_preview: Se deve mostrar preview durante processamento.

        Returns:
            Resultado completo da análise.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

        logger.info(f"Iniciando processamento: {video_path.name}")

        # Abre o vídeo
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

        # Obtém propriedades do vídeo
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        logger.info(f"Vídeo: {width}x{height}, {fps:.2f} FPS, {total_frames} frames, {duration:.2f}s")

        # Configura o vídeo de saída
        writer = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*settings.OUTPUT_VIDEO_CODEC)
            output_fps = settings.OUTPUT_VIDEO_FPS or fps
            writer = cv2.VideoWriter(
                str(output_path), fourcc, output_fps, (width, height)
            )
            logger.info(f"Salvando vídeo processado em: {output_path}")

        # Inicializa os analisadores
        for analyzer in self._analyzers:
            analyzer.setup()

        # Resultado da análise
        result = VideoAnalysisResult(
            video_path=str(video_path),
            total_frames=total_frames,
            processed_frames=0,
            fps=fps,
            duration_seconds=duration
        )

        # Processa os frames
        frame_number = 0
        processed_count = 0

        with tqdm(total=total_frames, desc="Processando", unit="frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_number += 1
                should_analyze = (frame_number % self._frame_sample_rate) == 0

                if should_analyze:
                    # Cria análise do frame
                    frame_analysis = FrameAnalysis(
                        frame_number=frame_number,
                        timestamp_ms=(frame_number / fps) * 1000 if fps > 0 else 0
                    )

                    # Aplica cada analisador
                    all_detections = []
                    for analyzer in self._analyzers:
                        detections = analyzer.analyze(frame, frame_number)
                        if analyzer.name == "face_detection":
                            for det in detections:
                                frame_analysis.add_face(det)
                        elif analyzer.name == "emotion_detection":
                            for det in detections:
                                frame_analysis.add_emotion(det)
                        elif analyzer.name == "activity_detection":
                            for det in detections:
                                frame_analysis.add_activity(det)
                        all_detections.extend(detections)

                    result.frame_analyses.append(frame_analysis)
                    processed_count += 1

                    # Desenha as detecções no frame
                    if writer or show_preview:
                        draw_all_detections(frame, frame_analysis.faces)
                        draw_all_emotions(frame, frame_analysis.emotions)
                        draw_all_activities(frame, frame_analysis.activities)
                        draw_frame_info(
                            frame,
                            frame_number,
                            frame_analysis.face_count,
                            fps
                        )

                # Escreve o frame no vídeo de saída
                if writer:
                    writer.write(frame)

                # Mostra preview
                if show_preview:
                    cv2.imshow("Video Analysis", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Processamento interrompido pelo usuário")
                        break

                pbar.update(1)

        # Finaliza
        result.processed_frames = processed_count
        cap.release()
        if writer:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()

        # Finaliza os analisadores
        for analyzer in self._analyzers:
            analyzer.teardown()

        logger.info(f"Processamento concluído: {processed_count} frames analisados")
        logger.info(f"Total de rostos detectados: {result.total_faces_detected}")
        logger.info(f"Total de emoções detectadas: {result.total_emotions_detected}")
        logger.info(f"Total de atividades detectadas: {result.total_activities_detected}")

        return result

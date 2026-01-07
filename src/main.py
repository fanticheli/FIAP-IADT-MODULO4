"""
Ponto de entrada principal da aplicação de análise de vídeo.
Tech Challenge FIAP - Fase 4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analyzers.face_analyzer_mp import FaceAnalyzerMediaPipe
from src.analyzers.emotion_analyzer import EmotionAnalyzer
from src.analyzers.activity_analyzer import ActivityAnalyzer
from src.config.settings import settings
from src.core.video_processor import VideoProcessor
from src.utils.logger import logger
from src.utils.report_generator import gerar_relatorio_pdf
from src.utils.frame_extractor import extrair_frames_anomalias


def parse_args() -> argparse.Namespace:
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Análise de Vídeo - Tech Challenge FIAP Fase 4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python -m src.main                      # Usa vídeo padrão
  python -m src.main -i video.mp4         # Especifica vídeo de entrada
  python -m src.main --preview            # Mostra preview durante processamento
  python -m src.main --confidence 0.3     # Detecta rostos mais distantes
        """
    )

    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=settings.VIDEOS_DIR / "video.mp4",
        help="Caminho do vídeo de entrada (padrão: videos/video.mp4)"
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Diretório de saída (padrão: resultado/{nome_video}/)"
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Mostra preview durante o processamento"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confiança mínima para detecção (0.0-1.0, use 0.3 para rostos distantes)"
    )

    return parser.parse_args()


def main() -> int:
    """Função principal da aplicação."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Análise de Vídeo - Tech Challenge FIAP Fase 4")
    logger.info("=" * 60)

    # Valida entrada
    if not args.input.exists():
        logger.error(f"Vídeo de entrada não encontrado: {args.input}")
        return 1

    # Define diretório de saída
    video_name = args.input.stem
    output_dir = args.output_dir or settings.OUTPUT_DIR / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Caminhos de saída
    output_video_path = output_dir / "video_analyzed.mp4"
    output_pdf_path = output_dir / "relatorio.pdf"

    logger.info(f"Vídeo de entrada: {args.input}")
    logger.info(f"Diretório de saída: {output_dir}")

    # Configura analisadores
    face_analyzer = FaceAnalyzerMediaPipe(min_detection_confidence=args.confidence)
    emotion_analyzer = EmotionAnalyzer()
    activity_analyzer = ActivityAnalyzer()

    # Processa o vídeo
    processor = VideoProcessor(analyzers=[face_analyzer, emotion_analyzer, activity_analyzer])

    try:
        result = processor.process(
            video_path=args.input,
            output_path=output_video_path,
            show_preview=args.preview
        )
    except Exception as e:
        logger.error(f"Erro durante processamento: {e}")
        return 1

    # Obtém resumo e anomalias
    summary = result.get_summary()
    anomalias = summary.get('anomalias', [])

    # Extrai frames das anomalias
    imagens_anomalias = []
    if anomalias:
        logger.info("Extraindo frames de anomalias...")
        imagens_anomalias = extrair_frames_anomalias(
            video_path=args.input,
            anomalias=anomalias,
            output_dir=output_dir,
            max_frames=3
        )

    # Gera relatório PDF
    gerar_relatorio_pdf(summary, output_pdf_path, imagens_anomalias)

    # Exibe resumo no console
    logger.info("=" * 60)
    logger.info("RESUMO DA ANÁLISE")
    logger.info("=" * 60)
    logger.info(f"  Frames processados: {summary.get('total_frames', 0)}")
    logger.info(f"  Rostos detectados: {summary.get('total_face_detections', 0)}")
    logger.info(f"  Frames com rostos: {summary.get('frames_with_faces', 0)}")
    logger.info(f"  Emoções detectadas: {summary.get('total_emotion_detections', 0)}")
    logger.info(f"  Atividades detectadas: {summary.get('total_activity_detections', 0)}")

    logger.info("=" * 60)
    logger.info(f"Relatório PDF: {output_pdf_path}")
    logger.info(f"Vídeo anotado: {output_video_path}")
    logger.info("Processamento finalizado com sucesso!")

    return 0


if __name__ == "__main__":
    sys.exit(main())

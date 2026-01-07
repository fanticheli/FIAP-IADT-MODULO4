"""
Gerador de relatorio PDF.
"""
from datetime import datetime
from pathlib import Path

from fpdf import FPDF


def gerar_relatorio_pdf(
    summary: dict,
    output_path: Path,
    imagens_anomalias: list[Path] = None
) -> None:
    """
    Gera relatorio PDF com os resultados da analise.
    """
    pdf = FPDF()
    pdf.add_page()

    # === CABECALHO ===
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 15, "Relatorio de Analise de Video", ln=True, align="C")

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, "Tech Challenge FIAP - Fase 4", ln=True, align="C")
    pdf.cell(0, 8, f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True, align="C")

    pdf.ln(5)

    # === INFORMACOES DO VIDEO ===
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 8, "Informacoes do Video", ln=True, fill=True)

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"  Arquivo: {summary.get('video_path', 'N/A')}", ln=True)
    pdf.cell(0, 6, f"  Total de frames: {summary.get('total_frames', 0)}", ln=True)
    pdf.cell(0, 6, f"  FPS: {summary.get('fps', 0)}", ln=True)
    pdf.cell(0, 6, f"  Duracao: {summary.get('duration_seconds', 0):.1f} segundos", ln=True)

    pdf.ln(8)

    # === SECAO 1: RECONHECIMENTO FACIAL ===
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(0, 100, 0)  # Verde escuro
    pdf.cell(0, 10, "1. Reconhecimento Facial", ln=True)
    pdf.set_text_color(0, 0, 0)

    pdf.set_font("Helvetica", "", 10)
    total_faces = summary.get('total_face_detections', 0)
    frames_with_faces = summary.get('frames_with_faces', 0)
    total_frames = summary.get('total_frames', 1)
    percentual = (frames_with_faces / total_frames) * 100 if total_frames > 0 else 0

    pdf.cell(0, 6, f"  Total de rostos detectados: {total_faces}", ln=True)
    pdf.cell(0, 6, f"  Frames com rostos: {frames_with_faces}", ln=True)
    pdf.cell(0, 6, f"  Percentual de frames com rostos: {percentual:.1f}%", ln=True)

    # Anomalias Faciais
    face_anomalies = summary.get('face_anomalies', [])
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, f"  Anomalias: {len(face_anomalies)}", ln=True)

    pdf.set_font("Helvetica", "", 10)
    if face_anomalies:
        for anomalia in face_anomalies[:5]:
            pdf.cell(0, 5, f"    - Frame {anomalia['frame']}: {anomalia['descricao']}", ln=True)
        if len(face_anomalies) > 5:
            pdf.cell(0, 5, f"    ... e mais {len(face_anomalies) - 5}", ln=True)

    pdf.ln(8)

    # === SECAO 2: ANALISE DE EMOCOES ===
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(0, 0, 150)  # Azul escuro
    pdf.cell(0, 10, "2. Analise de Expressoes Emocionais", ln=True)
    pdf.set_text_color(0, 0, 0)

    pdf.set_font("Helvetica", "", 10)
    total_emotions = summary.get('total_emotion_detections', 0)
    emotion_counts = summary.get('emotion_counts', {})

    pdf.cell(0, 6, f"  Total de emocoes detectadas: {total_emotions}", ln=True)

    if emotion_counts:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, "  Distribuicao de Emocoes:", ln=True)

        pdf.set_font("Helvetica", "", 10)
        sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
        for emotion, count in sorted_emotions:
            pct = (count / total_emotions * 100) if total_emotions > 0 else 0
            pdf.cell(0, 5, f"    - {emotion}: {count} ({pct:.1f}%)", ln=True)

    # Anomalias de Emoção
    emotion_anomalies = summary.get('emotion_anomalies', [])
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, f"  Anomalias: {len(emotion_anomalies)}", ln=True)

    pdf.set_font("Helvetica", "", 10)
    if emotion_anomalies:
        for anomalia in emotion_anomalies[:5]:
            pdf.cell(0, 5, f"    - Frame {anomalia['frame']}: {anomalia['descricao']}", ln=True)
        if len(emotion_anomalies) > 5:
            pdf.cell(0, 5, f"    ... e mais {len(emotion_anomalies) - 5}", ln=True)

    pdf.ln(8)

    # === SECAO 3: DETECCAO DE ATIVIDADES ===
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(150, 100, 0)  # Laranja escuro
    pdf.cell(0, 10, "3. Deteccao de Atividades", ln=True)
    pdf.set_text_color(0, 0, 0)

    total_activities = summary.get('total_activity_detections', 0)
    activity_counts = summary.get('activity_counts', {})

    if total_activities > 0:
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"  Total de atividades detectadas: {total_activities}", ln=True)

        if activity_counts:
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 6, "  Distribuicao de Atividades:", ln=True)

            pdf.set_font("Helvetica", "", 10)
            sorted_activities = sorted(activity_counts.items(), key=lambda x: x[1], reverse=True)
            for activity, count in sorted_activities:
                pct = (count / total_activities * 100) if total_activities > 0 else 0
                pdf.cell(0, 5, f"    - {activity}: {count} ({pct:.1f}%)", ln=True)

        # Anomalias de Atividade
        activity_anomalies = summary.get('activity_anomalies', [])
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, f"  Anomalias: {len(activity_anomalies)}", ln=True)

        pdf.set_font("Helvetica", "", 10)
        if activity_anomalies:
            for anomalia in activity_anomalies[:5]:
                pdf.cell(0, 5, f"    - Frame {anomalia['frame']}: {anomalia['descricao']}", ln=True)
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 6, "  (Em desenvolvimento)", ln=True)

    # === IMAGENS DAS ANOMALIAS ===
    if imagens_anomalias:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Anexo: Frames com Anomalias", ln=True)

        for img_path in imagens_anomalias:
            if img_path.exists():
                pdf.ln(5)
                pdf.set_font("Helvetica", "", 9)
                pdf.cell(0, 5, f"Frame: {img_path.stem}", ln=True)
                pdf.image(str(img_path), x=10, w=190)
                pdf.ln(3)

    pdf.output(output_path)

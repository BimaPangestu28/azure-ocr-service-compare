"""
OCR Comparison Tool - Azure OpenAI vs Content Understanding

Streamlit UI untuk membandingkan hasil OCR dari Azure OpenAI Multimodal
dan Azure Content Understanding dengan live comparison.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from services import AzureContentUnderstanding, AzureOpenAIOCR, MistralOCR


def convert_pdf_to_image(pdf_bytes: bytes, dpi: int = 200) -> bytes:
    """
    Convert first page of PDF to PNG image.

    @param pdf_bytes - PDF file as bytes
    @param dpi - Resolution for rendering
    @returns PNG image as bytes
    """
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = pdf_document[0]  # First page

    # Render page to image
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)

    # Convert to PNG bytes
    img_bytes = pix.tobytes("png")
    pdf_document.close()

    return img_bytes


def convert_pdf_to_images(pdf_bytes: bytes, dpi: int = 200) -> list[bytes]:
    """
    Convert all pages of PDF to PNG images.

    @param pdf_bytes - PDF file as bytes
    @param dpi - Resolution for rendering
    @returns List of PNG images as bytes
    """
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []

    for page in pdf_document:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        images.append(pix.tobytes("png"))

    pdf_document.close()
    return images

load_dotenv()


def initialize_services():
    """Initialize Azure services dari environment variables."""
    openai_service = None
    mistral_service = None
    content_service = None
    errors = []

    # Azure OpenAI
    openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai_key = os.getenv("AZURE_OPENAI_API_KEY")
    openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    openai_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    if openai_endpoint and openai_key:
        openai_service = AzureOpenAIOCR(
            endpoint=openai_endpoint,
            api_key=openai_key,
            deployment_name=openai_deployment,
            api_version=openai_version
        )

    # Mistral Document AI
    mistral_endpoint = os.getenv("MISTRAL_ENDPOINT")
    mistral_key = os.getenv("MISTRAL_API_KEY")
    mistral_model = os.getenv("MISTRAL_MODEL", "mistral-document-ai-2505")

    if mistral_endpoint and mistral_key:
        mistral_service = MistralOCR(
            endpoint=mistral_endpoint,
            api_key=mistral_key,
            model=mistral_model
        )

    # Azure Content Understanding
    cu_endpoint = os.getenv("AZURE_CU_ENDPOINT")
    cu_key = os.getenv("AZURE_CU_SUBSCRIPTION_KEY")
    cu_analyzer = os.getenv("AZURE_CU_ANALYZER_ID")
    cu_version = os.getenv("AZURE_CU_API_VERSION", "2025-05-01-preview")

    if cu_endpoint and cu_key and cu_analyzer:
        content_service = AzureContentUnderstanding(
            endpoint=cu_endpoint,
            subscription_key=cu_key,
            analyzer_id=cu_analyzer,
            api_version=cu_version
        )

    return openai_service, mistral_service, content_service, errors


def calculate_accuracy(text1: str, text2: str) -> float:
    """
    Hitung similarity antara dua text menggunakan Jaccard similarity.

    @param text1 - Text pertama
    @param text2 - Text kedua
    @returns Similarity score 0.0-1.0
    """
    if not text1 or not text2:
        return 0.0

    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union)


def format_comparison_metrics(llm_result, content_result, llm_name="LLM"):
    """Format metrics untuk comparison table."""
    llm_confidence = f"{llm_result.confidence_score:.2%}" if llm_result.confidence_score is not None else "N/A"
    content_confidence = f"{content_result.confidence_score:.2%}" if content_result.confidence_score is not None else "N/A"

    data = {
        "Metric": [
            "Processing Time",
            "Confidence Score",
            "Word Count",
            "Character Count"
        ],
        llm_name: [
            f"{llm_result.processing_time_ms:.2f} ms",
            llm_confidence,
            str(len(llm_result.text.split())),
            str(len(llm_result.text))
        ],
        "Content Understanding": [
            f"{content_result.processing_time_ms:.2f} ms",
            content_confidence,
            str(len(content_result.text.split())),
            str(len(content_result.text))
        ]
    }
    return pd.DataFrame(data)


def display_results(llm_result, content_result, ground_truth="", llm_name="LLM"):
    """Display comparison results."""
    st.header("Comparison Results")

    metric_cols = st.columns(4)

    speed_winner = "LLM" if llm_result.processing_time_ms < content_result.processing_time_ms else "Content"
    speed_diff = abs(llm_result.processing_time_ms - content_result.processing_time_ms)

    with metric_cols[0]:
        st.metric(
            f"{llm_name} Speed",
            f"{llm_result.processing_time_ms:.0f}ms",
            delta=f"-{speed_diff:.0f}ms faster" if speed_winner == "LLM" else f"+{speed_diff:.0f}ms slower",
            delta_color="normal" if speed_winner == "LLM" else "inverse"
        )

    with metric_cols[1]:
        st.metric(
            "Content Understanding Speed",
            f"{content_result.processing_time_ms:.0f}ms",
            delta=f"-{speed_diff:.0f}ms faster" if speed_winner == "Content" else f"+{speed_diff:.0f}ms slower",
            delta_color="normal" if speed_winner == "Content" else "inverse"
        )

    # Handle confidence - might be None for some models
    llm_conf = llm_result.confidence_score
    content_conf = content_result.confidence_score

    if llm_conf is not None and content_conf is not None:
        conf_winner = "LLM" if llm_conf > content_conf else "Content"
        conf_diff = abs(llm_conf - content_conf)

        with metric_cols[2]:
            st.metric(
                f"{llm_name} Confidence",
                f"{llm_conf:.1%}",
                delta=f"+{conf_diff:.1%}" if conf_winner == "LLM" else f"-{conf_diff:.1%}",
                delta_color="normal" if conf_winner == "LLM" else "inverse"
            )

        with metric_cols[3]:
            st.metric(
                "Content Confidence",
                f"{content_conf:.1%}",
                delta=f"+{conf_diff:.1%}" if conf_winner == "Content" else f"-{conf_diff:.1%}",
                delta_color="normal" if conf_winner == "Content" else "inverse"
            )
    else:
        with metric_cols[2]:
            st.metric(
                f"{llm_name} Confidence",
                "N/A" if llm_conf is None else f"{llm_conf:.1%}"
            )

        with metric_cols[3]:
            st.metric(
                "Content Confidence",
                "N/A" if content_conf is None else f"{content_conf:.1%}"
            )

    st.subheader("Detailed Metrics")
    metrics_df = format_comparison_metrics(llm_result, content_result, llm_name)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    if ground_truth:
        st.subheader("Accuracy vs Ground Truth")

        acc_col1, acc_col2 = st.columns(2)

        llm_accuracy = calculate_accuracy(llm_result.text, ground_truth)
        content_accuracy = calculate_accuracy(content_result.text, ground_truth)

        with acc_col1:
            st.metric(f"{llm_name} Accuracy", f"{llm_accuracy:.1%}")
            st.progress(llm_accuracy)

        with acc_col2:
            st.metric("Content Understanding Accuracy", f"{content_accuracy:.1%}")
            st.progress(content_accuracy)

    similarity = calculate_accuracy(llm_result.text, content_result.text)
    st.subheader("Result Similarity")
    st.metric(
        "Similarity between results",
        f"{similarity:.1%}",
        help="Jaccard similarity antara hasil OCR kedua service"
    )
    st.progress(similarity)

    st.divider()
    st.header("Extracted Text")

    text_col1, text_col2 = st.columns(2)

    with text_col1:
        st.subheader(f"{llm_name} Result")
        st.text_area(
            "Extracted text:",
            value=llm_result.text,
            height=300,
            key="llm_text_result"
        )

        # Show token usage for OpenAI, annotations for Mistral
        if hasattr(llm_result, 'token_usage') and llm_result.token_usage:
            with st.expander("Token Usage"):
                st.json(llm_result.token_usage)
        if hasattr(llm_result, 'annotations') and llm_result.annotations:
            with st.expander("Annotations"):
                st.json(llm_result.annotations)

    with text_col2:
        st.subheader("Content Understanding Result")
        st.text_area(
            "Extracted text:",
            value=content_result.text,
            height=300,
            key="content_text_result"
        )

        if hasattr(content_result, 'fields') and content_result.fields:
            with st.expander("Extracted Fields"):
                for field_name, field_data in content_result.fields.items():
                    st.markdown(f"**{field_name}:** {field_data.get('value', 'N/A')} (confidence: {field_data.get('confidence', 0):.1%})")

        with st.expander("Raw Response"):
            st.json(content_result.raw_response)

    st.divider()
    st.header("Summary")

    llm_accuracy = calculate_accuracy(llm_result.text, ground_truth) if ground_truth else None
    content_accuracy = calculate_accuracy(content_result.text, ground_truth) if ground_truth else None

    # Determine confidence winner
    llm_conf = llm_result.confidence_score
    content_conf = content_result.confidence_score

    if llm_conf is not None and content_conf is not None:
        conf_winner = llm_name if llm_conf > content_conf else "Content Understanding"
        conf_diff = f"{abs(llm_conf - content_conf):.1%}"
    else:
        conf_winner = "N/A"
        conf_diff = "N/A"

    summary_data = {
        "Category": ["Speed", "Confidence", "Accuracy (if ground truth)"],
        "Winner": [
            llm_name if llm_result.processing_time_ms < content_result.processing_time_ms else "Content Understanding",
            conf_winner,
            (llm_name if llm_accuracy > content_accuracy else "Content Understanding") if ground_truth else "N/A"
        ],
        "Difference": [
            f"{speed_diff:.2f}ms",
            conf_diff,
            f"{abs(llm_accuracy - content_accuracy):.1%}" if ground_truth else "N/A"
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


def main():
    st.set_page_config(
        page_title="OCR Comparison Tool",
        page_icon="🔍",
        layout="wide"
    )

    st.title("OCR Comparison Tool")
    st.markdown("**LLM Multimodal** vs **Azure Content Understanding** - Live Comparison")

    openai_service, mistral_service, content_service, errors = initialize_services()

    with st.sidebar:
        st.header("Configuration")

        # LLM Model Selection
        st.subheader("LLM Model")
        available_llms = []
        if openai_service:
            available_llms.append("Azure OpenAI (GPT-4o)")
        if mistral_service:
            available_llms.append("Mistral Document AI")

        if not available_llms:
            st.error("❌ No LLM configured")
            selected_llm = None
        else:
            selected_llm = st.radio(
                "Pilih model LLM:",
                available_llms,
                horizontal=True
            )

        st.divider()

        st.subheader("Ground Truth (Optional)")
        ground_truth = st.text_area(
            "Masukkan text asli untuk menghitung akurasi:",
            height=150,
            help="Opsional: digunakan untuk menghitung akurasi OCR"
        )

        st.divider()

        st.subheader("Service Status")
        st.markdown(f"**OpenAI:** {'✅ Ready' if openai_service else '❌ Not configured'}")
        st.markdown(f"**Mistral:** {'✅ Ready' if mistral_service else '❌ Not configured'}")
        st.markdown(f"**Content Understanding:** {'✅ Ready' if content_service else '❌ Not configured'}")

        with st.expander("Debug: Environment Variables"):
            st.code(f"""
AZURE_OPENAI_ENDPOINT: {'✅ Set' if os.getenv('AZURE_OPENAI_ENDPOINT') else '❌ Missing'}
AZURE_OPENAI_API_KEY: {'✅ Set' if os.getenv('AZURE_OPENAI_API_KEY') else '❌ Missing'}
AZURE_OPENAI_DEPLOYMENT_NAME: {os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4o (default)')}

MISTRAL_ENDPOINT: {'✅ Set' if os.getenv('MISTRAL_ENDPOINT') else '❌ Missing'}
MISTRAL_API_KEY: {'✅ Set' if os.getenv('MISTRAL_API_KEY') else '❌ Missing'}
MISTRAL_MODEL: {os.getenv('MISTRAL_MODEL', 'mistral-document-ai-2505 (default)')}

AZURE_CU_ENDPOINT: {'✅ Set' if os.getenv('AZURE_CU_ENDPOINT') else '❌ Missing'}
AZURE_CU_SUBSCRIPTION_KEY: {'✅ Set' if os.getenv('AZURE_CU_SUBSCRIPTION_KEY') else '❌ Missing'}
AZURE_CU_ANALYZER_ID: {'✅ Set' if os.getenv('AZURE_CU_ANALYZER_ID') else '❌ Missing'}
            """)

        st.divider()
        st.caption("Upload image untuk auto-run OCR comparison")

    uploaded_file = st.file_uploader(
        "Upload file untuk OCR (auto-run)",
        type=["png", "jpg", "jpeg", "tiff", "bmp", "pdf"],
        help="OCR akan berjalan otomatis setelah upload (support image & PDF)"
    )

    if uploaded_file:
        file_bytes = uploaded_file.read()
        is_pdf = uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf")

        col_preview, col_info = st.columns([2, 1])

        with col_preview:
            if is_pdf:
                st.markdown("**PDF Preview (Page 1):**")
                try:
                    preview_img = convert_pdf_to_image(file_bytes, dpi=150)
                    st.image(preview_img, caption=f"PDF: {uploaded_file.name}", use_container_width=True)
                except Exception:
                    st.markdown(f"📄 **{uploaded_file.name}**")
                    st.info("PDF preview not available")
            else:
                image = Image.open(BytesIO(file_bytes))
                st.image(image, caption="Uploaded Image", use_container_width=True)

        with col_info:
            st.markdown("**File Info:**")
            st.markdown(f"- Name: {uploaded_file.name}")
            st.markdown(f"- Type: {'PDF' if is_pdf else 'Image'}")
            if is_pdf:
                try:
                    pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
                    st.markdown(f"- Pages: {len(pdf_doc)}")
                    pdf_doc.close()
                except Exception:
                    pass
            else:
                image = Image.open(BytesIO(file_bytes))
                st.markdown(f"- Format: {image.format or 'Unknown'}")
                st.markdown(f"- Size: {image.size[0]}x{image.size[1]}")
                st.markdown(f"- Mode: {image.mode}")
            st.markdown(f"- File size: {len(file_bytes) / 1024:.2f} KB")

        st.divider()

        if is_pdf:
            mime_type = "application/pdf"
        elif uploaded_file.type:
            mime_type = uploaded_file.type
        else:
            mime_type = "image/png"

        # Determine which LLM service to use
        use_mistral = selected_llm == "Mistral Document AI" if selected_llm else False
        llm_service = mistral_service if use_mistral else openai_service
        llm_name = "Mistral Document AI" if use_mistral else "Azure OpenAI (GPT-4o)"

        # Setup UI columns dengan placeholders
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(llm_name)
            llm_status = st.empty()
            llm_detail = st.empty()

        with col2:
            st.subheader("Azure Content Understanding")
            content_status = st.empty()
            content_detail = st.empty()

        # Initial status
        if llm_service:
            llm_status.info("🔄 Processing...")
        else:
            llm_status.error("❌ Not configured")

        if content_service:
            content_status.info("🔄 Processing...")
        else:
            content_status.error("❌ Not configured")

        # Results storage
        llm_response = {"success": False, "error": "Not configured"}
        content_response = {"success": False, "error": "Not configured"}

        def run_llm():
            """Run LLM OCR (OpenAI or Mistral)."""
            try:
                if use_mistral:
                    result = mistral_service.extract_text(
                        file_bytes=file_bytes,
                        file_type=mime_type,
                        timeout_seconds=300  # 5 minutes for large PDFs
                    )
                else:
                    result = openai_service.extract_text(
                        image_bytes=file_bytes,
                        image_type=mime_type
                    )
                return {"success": True, "result": result}
            except Exception as e:
                return {"success": False, "error": str(e)}

        def run_content():
            """Run Content Understanding OCR."""
            try:
                result = content_service.extract_text(
                    image_bytes=file_bytes,
                    timeout_seconds=120
                )
                return {"success": True, "result": result}
            except Exception as e:
                return {"success": False, "error": str(e)}

        # Run in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}

            if llm_service:
                futures["llm"] = executor.submit(run_llm)
            if content_service:
                futures["content"] = executor.submit(run_content)

            # Wait and update UI as each completes
            import time as time_module
            while futures:
                for name in list(futures.keys()):
                    future = futures[name]
                    if future.done():
                        response = future.result()
                        if name == "llm":
                            llm_response = response
                            if response["success"]:
                                llm_status.success(f"✅ Done: {response['result'].processing_time_ms:.0f}ms")
                            else:
                                llm_status.error("❌ Failed")
                                llm_detail.error(response["error"])
                        else:
                            content_response = response
                            if response["success"]:
                                content_status.success(f"✅ Done: {response['result'].processing_time_ms:.0f}ms")
                            else:
                                content_status.error("❌ Failed")
                                content_detail.error(response["error"])
                        del futures[name]
                time_module.sleep(0.1)

        llm_result = llm_response.get("result") if llm_response["success"] else None
        content_result = content_response.get("result") if content_response["success"] else None

        if llm_result and content_result:
            st.divider()
            display_results(llm_result, content_result, ground_truth, llm_name)

        elif llm_result or content_result:
            st.divider()
            st.header("Partial Results")

            if llm_result:
                st.subheader(f"{llm_name} Result")
                st.text_area(
                    "Extracted text:",
                    value=llm_result.text,
                    height=300,
                    key="llm_partial"
                )

            if content_result:
                st.subheader("Content Understanding Result")
                st.text_area(
                    "Extracted text:",
                    value=content_result.text,
                    height=300,
                    key="content_partial"
                )

    else:
        st.info("👆 Upload an image to automatically start OCR comparison")

        st.markdown("""
        ### Features:
        - **Auto-run**: OCR langsung jalan setelah upload file
        - **PDF & Image Support**: Support PNG, JPG, JPEG, TIFF, BMP, dan PDF
        - **Parallel Processing**: Kedua service berjalan bersamaan
        - **Live Status**: Lihat progress real-time dari masing-masing service
        - **Speed Comparison**: Bandingkan waktu proses kedua service
        - **Confidence Score**: Lihat confidence score dari masing-masing service
        - **Accuracy Measurement**: Bandingkan dengan ground truth (optional)

        ### How to Use:
        1. Setup credentials di `.env` file (copy dari `.env.example`)
        2. Upload gambar atau PDF yang berisi text
        3. OCR akan berjalan otomatis dan menampilkan hasil perbandingan
        """)


if __name__ == "__main__":
    main()

"""
Mistral Document AI OCR Service

Service untuk melakukan OCR menggunakan Mistral Document AI via Azure.
"""

import base64
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests


@dataclass
class OCRResult:
    """Hasil OCR dengan metadata performa."""

    text: str
    processing_time_ms: float
    confidence_score: Optional[float]
    annotations: dict
    raw_response: Optional[dict] = None


class MistralOCR:
    """
    OCR menggunakan Mistral Document AI via Azure.

    @param endpoint - Azure endpoint URL
    @param api_key - Azure API key
    @param model - Model name (default: mistral-document-ai-2505)
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str = "mistral-document-ai-2505"
    ):
        if not endpoint:
            raise ValueError("Mistral endpoint is required")

        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.model = model

        # Check if endpoint already includes the OCR path
        if "/providers/mistral/azure/ocr" in self.endpoint:
            self.ocr_url = self.endpoint
        else:
            self.ocr_url = f"{self.endpoint}/providers/mistral/azure/ocr"

    def extract_text(
        self,
        file_bytes: bytes,
        file_type: str = "application/pdf",
        timeout_seconds: int = 300
    ) -> OCRResult:
        """
        Extract text dari document menggunakan Mistral Document AI.

        @param file_bytes - File dalam bentuk bytes
        @param file_type - MIME type (application/pdf, image/png, etc.)
        @param timeout_seconds - Timeout in seconds (default 300 = 5 minutes)
        @returns OCRResult dengan text, waktu proses, dan annotations
        """
        base64_data = base64.b64encode(file_bytes).decode("utf-8")

        payload = {
            "model": self.model,
            "document": {
                "type": "document_url",
                "document_url": f"data:{file_type};base64,{base64_data}"
            },
            "include_image_base64": False
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        start_time = time.perf_counter()

        response = requests.post(
            self.ocr_url,
            headers=headers,
            json=payload,
            timeout=timeout_seconds
        )
        response.raise_for_status()

        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000

        try:
            result = response.json()
        except Exception:
            result = {"error": "Failed to parse response", "text": response.text}

        if result is None:
            result = {}

        extracted_text, annotations = self._parse_result(result)

        return OCRResult(
            text=extracted_text,
            processing_time_ms=processing_time_ms,
            confidence_score=None,  # Mistral doesn't provide confidence
            annotations=annotations,
            raw_response=result
        )

    def _parse_result(self, result: dict) -> tuple[str, dict]:
        """
        Parse result dari Mistral OCR API.

        @param result - Raw API result
        @returns Tuple of (extracted_text, annotations)
        """
        extracted_text = ""
        annotations = {}

        if not result:
            return extracted_text, annotations

        # Extract dari pages jika ada
        pages = result.get("pages") or []
        text_parts = []

        for page in pages:
            if page and isinstance(page, dict):
                if "markdown" in page:
                    text_parts.append(page["markdown"])
                elif "text" in page:
                    text_parts.append(page["text"])

        if text_parts:
            extracted_text = "\n\n".join(text_parts)

        # Extract annotations jika ada
        if result.get("document_annotation"):
            annotations = result["document_annotation"]
            # Jika ada full_text di annotations, gunakan itu
            if isinstance(annotations, dict) and "full_text" in annotations:
                extracted_text = annotations["full_text"]

        # Fallback ke choices jika format berbeda
        choices = result.get("choices") or []
        if choices and not extracted_text:
            first_choice = choices[0] if choices else {}
            if isinstance(first_choice, dict):
                message = first_choice.get("message") or {}
                content = message.get("content", "")
                if content:
                    extracted_text = content

        return extracted_text, annotations

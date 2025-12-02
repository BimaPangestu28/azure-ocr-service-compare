"""
Azure OpenAI Multimodal OCR Service

Service untuk melakukan OCR menggunakan Azure OpenAI GPT-4o/GPT-4 Vision.
"""

import base64
import time
from dataclasses import dataclass
from typing import Optional

from openai import AzureOpenAI


@dataclass
class OCRResult:
    """Hasil OCR dengan metadata performa."""

    text: str
    processing_time_ms: float
    confidence_score: float
    token_usage: dict
    raw_response: Optional[dict] = None


class AzureOpenAIOCR:
    """
    OCR menggunakan Azure OpenAI Multimodal (GPT-4o/GPT-4 Vision).

    @param endpoint - Azure OpenAI endpoint URL
    @param api_key - Azure OpenAI API key
    @param deployment_name - Nama deployment model (e.g., gpt-4o)
    @param api_version - API version
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment_name: str,
        api_version: str = "2024-02-15-preview"
    ):
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )
        self.deployment_name = deployment_name

    def encode_image_to_base64(self, image_bytes: bytes) -> str:
        """Encode image bytes ke base64 string."""
        return base64.b64encode(image_bytes).decode("utf-8")

    def extract_text(
        self,
        image_bytes: bytes,
        image_type: str = "image/png",
        prompt: str = None
    ) -> OCRResult:
        """
        Extract text dari image/PDF menggunakan Azure OpenAI Vision.

        @param image_bytes - File dalam bentuk bytes (image atau PDF)
        @param image_type - MIME type (image/png, image/jpeg, application/pdf, etc.)
        @param prompt - Custom prompt untuk extraction (optional)
        @returns OCRResult dengan text, waktu proses, dan confidence score
        """
        base64_data = self.encode_image_to_base64(image_bytes)

        if prompt is None:
            if image_type == "application/pdf":
                prompt = "Extract all text from this PDF document. Return only the extracted text, maintaining the original layout and structure as much as possible."
            else:
                prompt = "Extract all text from this image. Return only the extracted text, maintaining the original layout and structure as much as possible."

        start_time = time.perf_counter()

        # Azure OpenAI GPT-4o accepts PDF via image_url with data URI
        user_content = [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image_type};base64,{base64_data}",
                    "detail": "high"
                }
            }
        ]

        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert OCR system. Extract text accurately and completely. Also provide a confidence score (0.0-1.0) for your extraction at the end in format: [CONFIDENCE: X.XX]"
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            max_tokens=4096
        )

        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000

        raw_text = response.choices[0].message.content

        confidence_score = self._extract_confidence_score(raw_text)
        cleaned_text = self._clean_text(raw_text)

        token_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

        return OCRResult(
            text=cleaned_text,
            processing_time_ms=processing_time_ms,
            confidence_score=confidence_score,
            token_usage=token_usage,
            raw_response=response.model_dump()
        )

    def _extract_confidence_score(self, text: str) -> float:
        """Extract confidence score dari response text."""
        import re

        pattern = r'\[CONFIDENCE:\s*([\d.]+)\]'
        match = re.search(pattern, text)

        if match:
            try:
                score = float(match.group(1))
                return min(max(score, 0.0), 1.0)
            except ValueError:
                pass

        return 0.85

    def _clean_text(self, text: str) -> str:
        """Remove confidence marker dari text."""
        import re

        cleaned = re.sub(r'\[CONFIDENCE:\s*[\d.]+\]', '', text)
        return cleaned.strip()

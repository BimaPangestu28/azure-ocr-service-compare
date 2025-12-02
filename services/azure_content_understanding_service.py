"""
Azure AI Content Understanding OCR Service

Service untuk melakukan OCR menggunakan Azure AI Content Understanding
dengan custom analyzer.
"""

import time
from dataclasses import dataclass
from typing import Any, Optional

import requests


@dataclass
class OCRResult:
    """Hasil OCR dengan metadata performa."""

    text: str
    processing_time_ms: float
    confidence_score: float
    fields: dict
    raw_response: Optional[dict] = None


class AzureContentUnderstanding:
    """
    OCR menggunakan Azure AI Content Understanding.

    @param endpoint - Azure Content Understanding endpoint URL
    @param subscription_key - Azure subscription key
    @param analyzer_id - ID analyzer yang sudah dikonfigurasi
    @param api_version - API version (default: 2025-05-01-preview)
    """

    def __init__(
        self,
        endpoint: str,
        subscription_key: str,
        analyzer_id: str,
        api_version: str = "2025-05-01-preview"
    ):
        self.endpoint = endpoint.rstrip("/")
        self.subscription_key = subscription_key
        self.analyzer_id = analyzer_id
        self.api_version = api_version
        self.headers = {
            "Ocp-Apim-Subscription-Key": subscription_key,
            "x-ms-useragent": "ocr-comparison-tool"
        }

    def _get_analyze_url(self) -> str:
        """Generate analyze endpoint URL."""
        return (
            f"{self.endpoint}/contentunderstanding/analyzers/{self.analyzer_id}:analyze"
            f"?api-version={self.api_version}&stringEncoding=utf16"
        )

    def _begin_analyze(self, image_bytes: bytes) -> requests.Response:
        """
        Mulai proses analisis image.

        @param image_bytes - Image dalam bentuk bytes
        @returns Response dengan operation-location header
        """
        headers = {
            **self.headers,
            "Content-Type": "application/octet-stream"
        }

        response = requests.post(
            url=self._get_analyze_url(),
            headers=headers,
            data=image_bytes
        )
        response.raise_for_status()
        return response

    def _begin_analyze_url(self, image_url: str) -> requests.Response:
        """
        Mulai proses analisis dari URL.

        @param image_url - URL image
        @returns Response dengan operation-location header
        """
        headers = {
            **self.headers,
            "Content-Type": "application/json"
        }

        response = requests.post(
            url=self._get_analyze_url(),
            headers=headers,
            json={"url": image_url}
        )
        response.raise_for_status()
        return response

    def _poll_result(
        self,
        response: requests.Response,
        timeout_seconds: int = 120,
        polling_interval_seconds: float = 1.0
    ) -> dict[str, Any]:
        """
        Poll hasil analisis sampai selesai.

        @param response - Response dari begin_analyze
        @param timeout_seconds - Timeout dalam detik
        @param polling_interval_seconds - Interval polling
        @returns Result dictionary
        """
        operation_location = response.headers.get("operation-location", "")
        if not operation_location:
            raise ValueError("Operation location not found in response headers.")

        start_time = time.time()

        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                raise TimeoutError(
                    f"Operation timed out after {timeout_seconds:.2f} seconds."
                )

            poll_response = requests.get(
                operation_location,
                headers=self.headers
            )
            poll_response.raise_for_status()

            result = poll_response.json()
            status = result.get("status", "").lower()

            if status == "succeeded":
                return result
            elif status == "failed":
                error_msg = result.get("error", {}).get("message", "Unknown error")
                raise RuntimeError(f"Analysis failed: {error_msg}")

            time.sleep(polling_interval_seconds)

    def extract_text(
        self,
        image_bytes: bytes,
        timeout_seconds: int = 120
    ) -> OCRResult:
        """
        Extract text dari image menggunakan Azure Content Understanding.

        @param image_bytes - Image dalam bentuk bytes
        @param timeout_seconds - Timeout untuk polling
        @returns OCRResult dengan text, waktu proses, dan confidence score
        """
        start_time = time.perf_counter()

        response = self._begin_analyze(image_bytes)
        result = self._poll_result(response, timeout_seconds=timeout_seconds)

        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000

        extracted_text, confidence_score, fields = self._parse_result(result)

        return OCRResult(
            text=extracted_text,
            processing_time_ms=processing_time_ms,
            confidence_score=confidence_score,
            fields=fields,
            raw_response=result
        )

    def extract_text_from_url(
        self,
        image_url: str,
        timeout_seconds: int = 120
    ) -> OCRResult:
        """
        Extract text dari image URL menggunakan Azure Content Understanding.

        @param image_url - URL image
        @param timeout_seconds - Timeout untuk polling
        @returns OCRResult dengan text, waktu proses, dan confidence score
        """
        start_time = time.perf_counter()

        response = self._begin_analyze_url(image_url)
        result = self._poll_result(response, timeout_seconds=timeout_seconds)

        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000

        extracted_text, confidence_score, fields = self._parse_result(result)

        return OCRResult(
            text=extracted_text,
            processing_time_ms=processing_time_ms,
            confidence_score=confidence_score,
            fields=fields,
            raw_response=result
        )

    def _parse_result(self, result: dict) -> tuple[str, float, dict]:
        """
        Parse result dari Content Understanding API.

        @param result - Raw API result
        @returns Tuple of (extracted_text, confidence_score, fields)
        """
        extracted_text = ""
        confidence_scores = []
        fields = {}

        analyze_result = result.get("result", {})

        # Extract content/text
        contents = analyze_result.get("contents", [])
        text_parts = []
        for content in contents:
            if "markdown" in content:
                text_parts.append(content["markdown"])
            elif "text" in content:
                text_parts.append(content["text"])

        extracted_text = "\n".join(text_parts)

        # Extract fields dan confidence
        result_fields = analyze_result.get("fields", {})
        for field_name, field_data in result_fields.items():
            if isinstance(field_data, dict):
                field_value = field_data.get("valueString", field_data.get("value", ""))
                field_confidence = field_data.get("confidence", 0.0)

                fields[field_name] = {
                    "value": field_value,
                    "confidence": field_confidence
                }

                if field_confidence:
                    confidence_scores.append(field_confidence)

        average_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores
            else 0.85
        )

        return extracted_text, average_confidence, fields

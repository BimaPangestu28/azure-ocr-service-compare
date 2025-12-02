# OCR Comparison Tool

Compare OCR results between **Azure OpenAI Multimodal (GPT-4o)** and **Azure AI Content Understanding**.

## Features

- **Speed Comparison**: Compare processing time between services
- **Confidence Score**: View confidence scores from each service
- **Accuracy Measurement**: Compare against ground truth (optional)
- **Text Similarity**: See how similar the OCR results are between services
- **Extracted Fields**: View structured fields from Content Understanding analyzer

## Prerequisites

- Python 3.9+
- Azure OpenAI resource with GPT-4o deployment
- Azure AI Content Understanding resource with configured analyzer

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env with your credentials
```

## Configuration

Edit `.env` file with your Azure credentials:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Azure Content Understanding Configuration
AZURE_CU_ENDPOINT=https://your-resource.services.ai.azure.com/
AZURE_CU_SUBSCRIPTION_KEY=your-subscription-key
AZURE_CU_ANALYZER_ID=your-analyzer-id
AZURE_CU_API_VERSION=2025-05-01-preview
```

## Usage

```bash
streamlit run app.py
```

Then:
1. Upload an image containing text
2. Click "Run OCR Comparison"
3. View and analyze the results

## Project Structure

```
asj/
├── app.py                                         # Streamlit UI
├── services/
│   ├── __init__.py
│   ├── azure_openai_service.py                    # Azure OpenAI GPT-4o OCR
│   └── azure_content_understanding_service.py     # Azure AI Content Understanding
├── requirements.txt
├── .env.example
└── README.md
```

## Metrics Compared

| Metric | Description |
|--------|-------------|
| Processing Time | Time to complete OCR (ms) |
| Confidence Score | Self-reported confidence (0-100%) |
| Word Count | Number of words extracted |
| Accuracy | Similarity to ground truth (if provided) |

## Azure Content Understanding

Azure AI Content Understanding uses custom analyzers. You need to:
1. Create an analyzer in Azure AI Foundry
2. Configure the analyzer for your document type
3. Use the analyzer ID in your `.env` file

API Reference: `{endpoint}/contentunderstanding/analyzers/{analyzer_id}:analyze`

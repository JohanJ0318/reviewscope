# Serving image: onnxruntime + tokenizers + FAISS. No torch, no transformers.
# Build after the pipeline has written artifacts/onnx and artifacts/index.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    REVIEWSCOPE_ARTIFACTS=/app/artifacts
WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .

COPY artifacts/onnx/classifier.onnx artifacts/onnx/classifier.int8.onnx artifacts/onnx/embedder.onnx artifacts/onnx/
COPY artifacts/onnx/classifier_tokenizer/ artifacts/onnx/classifier_tokenizer/
COPY artifacts/onnx/embedder_tokenizer/ artifacts/onnx/embedder_tokenizer/
COPY artifacts/index/ artifacts/index/

USER 10001
EXPOSE 8000
CMD ["uvicorn", "reviewscope.serve:app", "--host", "0.0.0.0", "--port", "8000"]

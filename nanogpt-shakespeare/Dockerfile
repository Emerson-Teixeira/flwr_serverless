FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for caching
RUN pip install --no-cache-dir "flwr[simulation]>=1.26.1" "torch>=2.0.0" --extra-index-url https://download.pytorch.org/whl/cpu numpy requests

# Copy app code
COPY pyproject.toml .
COPY README.md .
COPY nanogpt_shakespeare/ nanogpt_shakespeare/

RUN pip install --no-cache-dir -e .

CMD ["flwr", "run", ".", "--stream"]

FROM python:3.11-slim

WORKDIR /app

# --- Dependency layer (cached until pyproject.toml or src/ changes) ----------
# src/ is needed during install so the editable-mode .pth file is created
# correctly and points to the real package layout.
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e .

# --- Source layer (fast copy; no reinstall unless pyproject.toml changed) ----
COPY scripts/ scripts/
COPY prompts/ prompts/
# Only the golden dataset JSON is needed at runtime — corpus chunks are not.
COPY data/golden_dataset_v1.0.0.json data/golden_dataset_v1.0.0.json

# Runtime secrets must be injected via `docker run -e` or a secrets manager.
# Empty defaults prevent SDK import-time crashes when the image is inspected.
ENV OPENAI_API_KEY=""
ENV ANTHROPIC_API_KEY=""
ENV SLACK_WEBHOOK_URL=""

# Threshold values for the regression detector; override at runtime if needed.
ENV REGRESSION_THRESHOLD_WARNING="0.03"
ENV REGRESSION_THRESHOLD_CRITICAL="0.08"

# Use ENTRYPOINT so `docker run <image> --dataset ... --output-dir ...` works.
ENTRYPOINT ["python", "scripts/run_eval_ci.py"]

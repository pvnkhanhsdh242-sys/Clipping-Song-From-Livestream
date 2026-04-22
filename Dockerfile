FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
        && apt-get install -y --no-install-recommends ffmpeg libsndfile1 git \
        && if apt-cache show libchromaprint-tools >/dev/null 2>&1; then \
                 apt-get install -y --no-install-recommends libchromaprint-tools; \
             elif apt-cache show chromaprint-tools >/dev/null 2>&1; then \
                 apt-get install -y --no-install-recommends chromaprint-tools; \
             else \
                 echo "Chromaprint tools package unavailable; fpcalc will not be installed in this image."; \
             fi \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "app.main", "--help"]

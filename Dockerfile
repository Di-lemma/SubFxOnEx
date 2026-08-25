FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN groupadd --gid 10001 extractor \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin extractor

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY --chown=extractor:extractor effect_extractor.py /app/effect_extractor.py
COPY --chown=extractor:extractor effect_ontology /app/effect_ontology

USER extractor

CMD ["python", "/app/effect_extractor.py"]

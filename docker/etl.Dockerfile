# Spark ETL in a Linux container, for machines where local Spark is awkward
# (on Windows it needs winutils.exe to write parquet). From the repo root:
#
#   docker build -f docker/etl.Dockerfile -t reviewscope-etl .
#   docker run --rm -m 7g -v "$PWD/data:/app/data" reviewscope-etl --raw data/raw/dataset.csv
FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir ".[etl]"

# The driver does all the work in local mode; give it most of the container's memory.
ENV PYSPARK_SUBMIT_ARGS="--driver-memory 5g pyspark-shell"
ENTRYPOINT ["python", "-m", "reviewscope.etl_spark"]

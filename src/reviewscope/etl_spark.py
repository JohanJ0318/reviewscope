"""Spark ETL: raw Steam reviews CSV -> cleaned, deduplicated, split parquet.

The full Kaggle dump is ~6.4M reviews (~2 GB CSV with multi-line quoted text).
Pandas can load it on a big laptop, but the dedupe, the per-game aggregates and
the multi-line CSV parse are where Spark earns its place, and the same job runs
unchanged on Databricks (see databricks/etl_job.py).

Splits are by game, not by review: every review of a test game is unseen during
training, so test accuracy measures generalisation to new games rather than
memorising one game's vocabulary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

# The Kaggle dump uses this placeholder for reviews with no text.
PLACEHOLDERS = ["Early Access Review", ""]


def read_raw(spark: SparkSession, path: str) -> DataFrame:
    return (
        spark.read.option("header", True)
        .option("multiLine", True)
        .option("escape", '"')
        .csv(path)
    )


def clean(raw: DataFrame, min_chars: int = 20) -> DataFrame:
    df = raw.select(
        F.col("app_id").cast("long").alias("app_id"),
        F.col("app_name"),
        F.trim(F.regexp_replace(F.col("review_text"), r"\s+", " ")).alias("text"),
        F.col("review_score").cast("int").alias("score"),
        F.col("review_votes").cast("int").alias("helpful"),
    )
    return (
        df.where(F.col("app_id").isNotNull() & F.col("score").isin(1, -1))
        .where(F.col("text").isNotNull() & ~F.col("text").isin(PLACEHOLDERS))
        .where(F.length("text") >= min_chars)
        # Copy-pasted reviews (memes, ASCII art) repeat within a game; keep one.
        .dropDuplicates(["app_id", "text"])
        .withColumn("label", (F.col("score") == 1).cast("int"))
        .withColumn("review_id", F.xxhash64("app_id", "text"))
        .withColumn("helpful", F.coalesce(F.col("helpful"), F.lit(0)))
        .drop("score")
    )


def assign_split(df: DataFrame, val_buckets: int = 1, test_buckets: int = 1) -> DataFrame:
    """Hash each game into one of 10 buckets: 8 train, 1 val, 1 test by default."""
    bucket = F.pmod(F.xxhash64("app_id"), F.lit(10))
    return df.withColumn(
        "split",
        F.when(bucket < 10 - val_buckets - test_buckets, "train")
        .when(bucket < 10 - test_buckets, "val")
        .otherwise("test"),
    )


def app_stats(df: DataFrame) -> DataFrame:
    return df.groupBy("app_id", "app_name", "split").agg(
        F.count("*").alias("n_reviews"),
        F.avg("label").alias("positive_rate"),
        F.avg("helpful").alias("helpful_rate"),
        F.expr("percentile_approx(length(text), 0.5)").alias("median_chars"),
    )


def run(spark: SparkSession, raw_path: str, out: str, min_chars: int = 20) -> dict:
    raw = read_raw(spark, raw_path)
    reviews = assign_split(clean(raw, min_chars)).cache()
    reviews.repartition("split").write.mode("overwrite").partitionBy("split").parquet(f"{out}/reviews")
    app_stats(reviews).write.mode("overwrite").parquet(f"{out}/apps")

    counts = {r["split"]: r["count"] for r in reviews.groupBy("split").count().collect()}
    report = {
        "raw_rows": raw.count(),
        "kept_rows": sum(counts.values()),
        "rows_by_split": counts,
        "apps_by_split": {r["split"]: r["count"] for r in reviews.select("app_id", "split").distinct().groupBy("split").count().collect()},
        "positive_rate": reviews.agg(F.avg("label")).first()[0],
    }
    reviews.unpersist()
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="data/raw/sample.csv", help="CSV file or glob")
    p.add_argument("--out", default="data/processed")
    p.add_argument("--min-chars", type=int, default=20)
    p.add_argument("--master", default="local[*]")
    args = p.parse_args()

    spark = (
        SparkSession.builder.master(args.master)
        .appName("reviewscope-etl")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )
    report = run(spark, args.raw, args.out, args.min_chars)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.out, "etl_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()

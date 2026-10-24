# Databricks notebook source
# MAGIC %md
# MAGIC # reviewscope ETL on Databricks
# MAGIC
# MAGIC Runs the same `etl_spark.run` as local mode, against the full Kaggle dump in a
# MAGIC Unity Catalog volume, on serverless compute (Databricks Free Edition works).
# MAGIC Upload the CSV to the volume first; gzipped is fine and a third of the upload.

# COMMAND ----------

# MAGIC %pip install "reviewscope[etl] @ git+https://github.com/JohanJ0318/reviewscope"

# COMMAND ----------

dbutils.widgets.text("raw", "/Volumes/workspace/reviewscope/raw/dataset.csv.gz")  # noqa: F821
dbutils.widgets.text("out", "/Volumes/workspace/reviewscope/processed")  # noqa: F821
raw = dbutils.widgets.get("raw")  # noqa: F821
out = dbutils.widgets.get("out")  # noqa: F821

# COMMAND ----------

import json
import time

from reviewscope.etl_spark import run

t0 = time.time()
report = run(spark, raw, out)  # noqa: F821  (spark is provided by the runtime)
report["seconds"] = round(time.time() - t0)
dbutils.fs.put(f"{out}/etl_report.json", json.dumps(report, indent=2), overwrite=True)  # noqa: F821
print(json.dumps(report, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC Download `processed/reviews` (parquet) and continue locally with `train_lora`,
# MAGIC or keep it in the volume and train on a GPU cluster.

# COMMAND ----------

display(spark.read.parquet(f"{out}/apps").orderBy("n_reviews", ascending=False).limit(20))  # noqa: F821

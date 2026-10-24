# Databricks notebook source
# MAGIC %md
# MAGIC # reviewscope ETL on Databricks
# MAGIC
# MAGIC Runs the same `etl_spark.run` as local mode, against the full Kaggle dump in a
# MAGIC Unity Catalog volume. Upload `steam_reviews.csv` to the volume first, and put
# MAGIC your GitHub user name in the `%pip install` line below.

# COMMAND ----------

# MAGIC %pip install "reviewscope @ git+https://github.com/REPLACE_WITH_YOUR_USER/reviewscope"

# COMMAND ----------

dbutils.widgets.text("raw", "/Volumes/main/reviewscope/raw/steam_reviews.csv")  # noqa: F821
dbutils.widgets.text("out", "/Volumes/main/reviewscope/processed")  # noqa: F821
raw = dbutils.widgets.get("raw")  # noqa: F821
out = dbutils.widgets.get("out")  # noqa: F821

# COMMAND ----------

import json

from reviewscope.etl_spark import run

report = run(spark, raw, out)  # noqa: F821  (spark is provided by the runtime)
print(json.dumps(report, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC Download `processed/reviews` (parquet) and continue locally with `train_lora`,
# MAGIC or keep it in the volume and train on a GPU cluster.

# COMMAND ----------

display(spark.read.parquet(f"{out}/apps").orderBy("n_reviews", ascending=False).limit(20))  # noqa: F821

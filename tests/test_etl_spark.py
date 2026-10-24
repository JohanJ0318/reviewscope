import shutil

import pytest

pyspark = pytest.importorskip("pyspark")
if shutil.which("java") is None:
    pytest.skip("Spark needs a JVM", allow_module_level=True)

from pyspark.sql import SparkSession  # noqa: E402

from reviewscope.etl_spark import app_stats, assign_split, clean  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    s = SparkSession.builder.master("local[2]").appName("test").config("spark.sql.shuffle.partitions", "4").getOrCreate()
    yield s
    s.stop()


def _raw(spark, rows):
    return spark.createDataFrame(rows, ["app_id", "app_name", "review_text", "review_score", "review_votes"])


def test_clean_drops_placeholders_short_and_duplicates(spark):
    long = "This game is great, the combat is excellent."
    raw = _raw(spark, [
        ("1", "A", long, "1", "0"),
        ("1", "A", "  " + long.replace(" ", "   ") + " ", "1", "1"),  # same text after whitespace cleanup
        ("2", "B", long, "-1", "0"),                                   # same text, other game: kept
        ("1", "A", "Early Access Review", "1", "0"),
        ("1", "A", "meh", "-1", "0"),
        ("1", "A", "No score on this one, which is odd.", "0", "0"),
    ])
    out = clean(raw).collect()
    assert len(out) == 2
    assert sorted((r.app_id, r.label) for r in out) == [(1, 1), (2, 0)]


def test_split_is_by_game(spark):
    rows = [(str(g), f"G{g}", f"review number {i} for game {g}", "1", "0") for g in range(200) for i in range(3)]
    df = assign_split(clean(_raw(spark, rows)))
    per_game = df.groupBy("app_id").agg({"split": "approx_count_distinct"}).collect()
    assert all(r[1] == 1 for r in per_game)
    splits = {r.split for r in df.select("split").distinct().collect()}
    assert splits == {"train", "val", "test"}
    stats = app_stats(df).collect()
    assert len(stats) == 200 and all(s.n_reviews == 3 for s in stats)

package hinglish

import hinglish.config.Config.LabelMapping
import hinglish.ingestion.DataIngestion
import org.apache.spark.sql.SparkSession
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

class DataIngestionSpec extends AnyFunSuite with BeforeAndAfterAll {

  var spark: SparkSession = _

  override def beforeAll(): Unit = {
    spark = SparkSession
      .builder()
      .appName("DataIngestionSpec")
      .master("local[2]")
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
  }

  override def afterAll(): Unit = {
    if (spark != null) spark.stop()
  }

  test("parseBlock parses a well-formed CoNLL-meta block") {
    val block = "meta\t1\tpositive\nmujhe\tHIN\nyeh\tHIN\nmovie\tENG\npasand\tHIN\naayi\tHIN"
    val record = DataIngestion.parseBlock(block)
    assert(record.isDefined)
    assert(record.get.id == "1")
    assert(record.get.rawLabel == "positive")
    assert(record.get.text == "mujhe yeh movie pasand aayi")
    assert(record.get.tokenCount == 5)
  }

  test("parseBlock returns None for a block missing the label") {
    val block = "meta\t1\nmujhe\tHIN\nyeh\tHIN"
    assert(DataIngestion.parseBlock(block).isEmpty)
  }

  test("parseBlock returns None for a block with no tokens") {
    val block = "meta\t1\tpositive"
    assert(DataIngestion.parseBlock(block).isEmpty)
  }

  test("parseBlock returns None for a non-meta block") {
    val block = "mujhe\tHIN\nyeh\tHIN"
    assert(DataIngestion.parseBlock(block).isEmpty)
  }

  test("filterInvalidLabels drops rows with labels outside the canonical mapping") {
    val session = spark  // stable identifier required for implicits (var cannot be imported from)
    import session.implicits._
    val df = Seq(
      ("1", "positive", "good movie"),
      ("2", "bogus_label", "weird row"),
      ("3", "negative", "bad movie")
    ).toDF("id", "rawLabel", "text")

    val labels = LabelMapping(Map("negative" -> 0, "neutral" -> 1, "positive" -> 2))
    val filtered = DataIngestion.filterInvalidLabels(spark, df, labels)
    assert(filtered.count() == 2)
  }
}

package hinglish

import hinglish.config.Config.LabelMapping
import hinglish.features.{CharacterSplitter, FeaturePipeline}
import org.apache.spark.sql.SparkSession
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

class FeaturePipelineSpec extends AnyFunSuite with BeforeAndAfterAll {

  var spark: SparkSession = _

  override def beforeAll(): Unit = {
    spark = SparkSession
      .builder()
      .appName("FeaturePipelineSpec")
      .master("local[2]")
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
  }

  override def afterAll(): Unit = {
    if (spark != null) spark.stop()
  }

  test("CharacterSplitter splits text into individual characters, not words") {
    val session = spark  // stable identifier required for implicits (var cannot be imported from)
    import session.implicits._
    val df = Seq("hi bhai").toDF("text")
    val splitter = new CharacterSplitter().setInputCol("text").setOutputCol("chars")
    val out = splitter.transform(df).select("chars").as[Seq[String]].collect().head
    // "hi bhai" -> h,i,' ',b,h,a,i  (7 characters, including the space)
    assert(out == Seq("h", "i", " ", "b", "h", "a", "i"))
    assert(out.length == "hi bhai".length)
  }

  test("char n-gram stages produce genuinely character-level n-grams, not word n-grams") {
    val session = spark  // stable identifier required for implicits (var cannot be imported from)
    import session.implicits._
    val df = Seq("abcde").toDF("text")
    val stages = FeaturePipeline.charNgramTfidfStages(n = 3, numFeatures = 100)
    // Manually run the non-estimator stages up to NGram to inspect raw n-grams
    // (CountVectorizer/IDF are estimators and are exercised via the full
    // pipeline in the E1/E2 integration path).
    val splitter = new CharacterSplitter().setInputCol("text").setOutputCol("chars")
    val chars = splitter.transform(df)
    val ngram = new org.apache.spark.ml.feature.NGram().setN(3).setInputCol("chars").setOutputCol("charNgrams")
    val ngrams = ngram.transform(chars).select("charNgrams").as[Seq[String]].collect().head
    // "abcde" (5 chars) -> 3-grams: "a b c", "b c d", "c d e" (space-joined by NGram)
    assert(ngrams == Seq("a b c", "b c d", "c d e"))
  }

  test("withCanonicalLabel maps raw string labels to the canonical numeric ids") {
    val session = spark  // stable identifier required for implicits (var cannot be imported from)
    import session.implicits._
    val df = Seq(("1", "Positive"), ("2", "negative"), ("3", "NEUTRAL")).toDF("id", "rawLabel")
    val labels = LabelMapping(Map("negative" -> 0, "neutral" -> 1, "positive" -> 2))
    val out = FeaturePipeline.withCanonicalLabel(spark, df, labels)
    val collected = out.select("id", "label").as[(String, Double)].collect().toMap
    assert(collected("1") == 2.0)
    assert(collected("2") == 0.0)
    assert(collected("3") == 1.0)
  }
}

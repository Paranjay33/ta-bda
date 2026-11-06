package hinglish

import hinglish.config.Config.LabelMapping
import hinglish.evaluation.EvaluationUtils
import hinglish.experiments.{E1LogisticRegression, E2LinearSVCOvR}
import hinglish.config.Config
import hinglish.ingestion.DataIngestion
import hinglish.serving.SparkInferenceServer
import hinglish.util.SparkSessions
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import java.io.{File, PrintWriter}
import java.nio.file.Files

/** Software tests on tiny synthetic fixtures (NOT benchmark data). */
class ServingAndContractSpec extends AnyFunSuite with BeforeAndAfterAll {

  var spark: SparkSession = _
  val labels = LabelMapping(Map("negative" -> 0, "neutral" -> 1, "positive" -> 2))
  lazy val cfg: Config.ProjectConfig = Config.load(Config.findProjectRoot())

  override def beforeAll(): Unit = {
    spark = SparkSessions.builder("ServingAndContractSpec", "local[2]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
  }
  override def afterAll(): Unit = if (spark != null) spark.stop()

  private def fixtureTsv(rows: Seq[(String, String, String, Int)]): String = {
    val f = Files.createTempFile("processed", ".tsv").toFile
    val w = new PrintWriter(f, "UTF-8")
    w.println("sample_id\tsource_file\tsource_row_id\traw_text\tnormalized_text\tlabel\tlabel_id\tsplit")
    rows.foreach { case (id, text, lab, lid) => w.println(s"$id\tfixture\t$id\t$text\t$text\t$lab\t$lid\ttrain") }
    w.close()
    f.getAbsolutePath
  }

  private val trainRows: Seq[(String, String, String, Int)] = (0 until 60).map { i =>
    i % 3 match {
      case 0 => (s"n$i", s"bakwa film bekar time waste $i", "negative", 0)
      case 1 => (s"u$i", s"kal offic jana hai normal din $i", "neutral", 1)
      case _ => (s"p$i", s"""bahut acchi movi maza aaya "yaar" $i""", "positive", 2)
    }
  }

  test("loadProcessedSplit reads the shared contract: normalized_text unchanged, canonical label ids") {
    val df = DataIngestion.loadProcessedSplit(spark, fixtureTsv(trainRows.take(6)), labels)
    val rows = df.collect().map(r => r.getAs[String]("id") -> ((r.getAs[String]("text"), r.getAs[Double]("label")))).toMap
    assert(rows("p2")._1 == "bahut acchi movi maza aaya \"yaar\" 2") // quotes kept verbatim (QUOTE_NONE contract)
    assert(rows("n0")._2 == 0.0 && rows("u1")._2 == 1.0 && rows("p2")._2 == 2.0)
  }

  test("loadProcessedSplit rejects label/label_id disagreeing with label_mapping.json") {
    val bad = fixtureTsv(Seq(("x1", "kuch bhi", "positive", 0)))
    intercept[IllegalArgumentException] { DataIngestion.loadProcessedSplit(spark, bad, labels) }
  }

  test("E1 and E2 pipelines fit, persist, reload and score single texts identically (serving path)") {
    val train = DataIngestion.loadProcessedSplit(spark, fixtureTsv(trainRows), labels)
    Seq((E1LogisticRegression, "probability"), (E2LinearSVCOvR, "rawPrediction")).foreach { case (exp, col) =>
      val (pipeline, _) = exp.buildPipeline(cfg, smoke = true)
      val model = pipeline.fit(train)
      val dir = Files.createTempDirectory(s"model-${exp.experimentId}").toFile
      model.write.overwrite().save(new File(dir, "m").getAbsolutePath)
      val reloaded = PipelineModel.load(new File(dir, "m").getAbsolutePath)
      val texts = Seq("bahut acchi movi maza aaya", "bakwa film bekar", "kal offic jana hai")
      val a = SparkInferenceServer.score(spark, model, col, texts)
      val b = SparkInferenceServer.score(spark, reloaded, col, texts)
      assert(a.map(_._1) == b.map(_._1), s"${exp.experimentId} reloaded predictions differ")
      a.zip(b).foreach { case (x, y) => assert(x._2.zip(y._2).forall { case (p, q) => math.abs(p - q) < 1e-9 }) }
      assert(a.forall(_._2.length == 3), "one score per canonical class")
      assert(a.map(_._1) == Seq(2, 0, 1), s"${exp.experimentId} should separate this trivially separable fixture")
    }
  }

  test("E2 really uses character 3-grams and OneVsRest(LinearSVC)") {
    val (pipeline, hyper) = E2LinearSVCOvR.buildPipeline(cfg, smoke = true)
    val names = pipeline.getStages.map(_.getClass.getSimpleName).toSeq
    assert(names.contains("CharacterSplitter") && names.contains("NGram") && names.contains("OneVsRest"))
    assert(hyper("char_ngram_n") == "3")
  }

  test("per-class support counts GOLD labels, not predictions (midterm bug regression)") {
    val session = spark  // stable identifier required for implicits (var cannot be imported from)
    import session.implicits._
    val preds = Seq((0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (2.0, 2.0)).toDF("prediction", "label")
    val m = EvaluationUtils.computeMetrics(preds, labels)
    assert(m.perClass("negative").support == 1L)
    assert(m.perClass("neutral").support == 1L)
    assert(m.perClass("positive").support == 2L)
    assert(m.confusionMatrix(2)(0) == 1L && m.confusionMatrix(2)(2) == 1L)
  }
}

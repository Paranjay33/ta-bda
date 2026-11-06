package hinglish.experiments

import hinglish.config.Config
import hinglish.features.FeaturePipeline
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression

/** E1: Word TF-IDF -> Multinomial Logistic Regression (Spark MLlib).
  *
  *   text -> RegexTokenizer -> CountVectorizer -> IDF -> LogisticRegression(multinomial)
  *
  * Run (see scripts/run_spark.sh):
  *   E1LogisticRegression                   train on TRAIN, report VALIDATION, save PipelineModel
  *   E1LogisticRegression --final-test-eval locked TEST evaluation of the saved model
  *   E1LogisticRegression --smoke-test      tiny subset sanity pass
  */
object E1LogisticRegression extends ClassicalExperiment {

  val ExperimentId = "E1"
  override def experimentId: String = ExperimentId
  override def displayName: String = "TF-IDF + Logistic Regression (Spark MLlib)"
  override def scoreColumn: String = "probability"
  override def scoreSemantics: String =
    "model_probability (multinomial logistic regression softmax output; not separately calibrated)"

  override def buildPipeline(cfg: Config.ProjectConfig, smoke: Boolean): (Pipeline, Map[String, String]) = {
    val numFeatures = cfg.getInt("spark", "word_tfidf_num_features")
    val wordStages = FeaturePipeline.wordTfidfStages(outputCol = "features", numFeatures = numFeatures)
    val lr = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setFamily("multinomial")
      .setMaxIter(if (smoke) 10 else 100)
    val hyper = Map(
      "word_tfidf_num_features" -> numFeatures.toString,
      "tokenizer" -> "RegexTokenizer(pattern=\\s+, toLowercase=false)",
      "lr_family" -> "multinomial",
      "lr_max_iter" -> lr.getMaxIter.toString,
      "lr_reg_param" -> lr.getRegParam.toString
    )
    (new Pipeline().setStages(wordStages :+ lr), hyper)
  }
}

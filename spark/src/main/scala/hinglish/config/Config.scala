package hinglish.config

import org.yaml.snakeyaml.Yaml

import java.io.{File, FileInputStream}
import java.util.{Map => JMap}
import scala.jdk.CollectionConverters._

/** Central configuration loader for the Spark/Scala branch.
  *
  * Reads the SAME `configs/label_mapping.json` file as the Python branch
  * (JSON is valid YAML, so SnakeYAML parses it directly) so the label -> id
  * mapping can never drift between branches. General hyperparameters/paths
  * come from the same `configs/config.yaml` used by Python.
  */
object Config {

  /** Canonical label mapping, e.g. Map("negative" -> 0, "neutral" -> 1, "positive" -> 2) */
  case class LabelMapping(labelToId: Map[String, Int]) {
    val idToLabel: Map[Int, String] = labelToId.map(_.swap)
    val numClasses: Int = labelToId.size
    def isValid(label: String): Boolean = labelToId.contains(label.trim.toLowerCase)
  }

  case class ProjectConfig(root: File, raw: JMap[String, Any], labels: LabelMapping) {

    private def navigate(keys: Seq[String]): Any = {
      var node: Any = raw
      for (k <- keys) {
        node match {
          case m: JMap[_, _] => node = m.asInstanceOf[JMap[String, Any]].get(k)
          case null =>
            throw new NoSuchElementException(s"Missing config key path: ${keys.mkString(".")}")
          case _ =>
            throw new NoSuchElementException(s"Unexpected non-map node at '$k' in path ${keys.mkString(".")}")
        }
      }
      node
    }

    def getString(keys: String*): String = navigate(keys).toString

    def getInt(keys: String*): Int = navigate(keys) match {
      case i: java.lang.Integer => i.intValue()
      case s: String            => s.toInt
      case other                => other.toString.toDouble.toInt
    }

    def getDouble(keys: String*): Double = navigate(keys) match {
      case d: java.lang.Double  => d.doubleValue()
      case i: java.lang.Integer => i.doubleValue()
      case s: String            => s.toDouble
      case other                => other.toString.toDouble
    }

    def path(keys: String*): String = new File(root, getString(keys: _*)).getCanonicalPath
  }

  def findProjectRoot(startDir: File = new File(".").getCanonicalFile): File = {
    sys.env.get("HINGLISH_PROJECT_ROOT") match {
      case Some(p) =>
        val root = new File(p).getCanonicalFile
        require(
          new File(root, "configs/config.yaml").exists(),
          s"HINGLISH_PROJECT_ROOT=$root does not contain configs/config.yaml"
        )
        root
      case None =>
        var dir: File = startDir
        while (dir != null && !new File(dir, "configs/config.yaml").exists()) {
          dir = dir.getParentFile
        }
        if (dir == null) {
          throw new IllegalStateException(
            "Could not locate project root (no configs/config.yaml found). " +
              "Set HINGLISH_PROJECT_ROOT explicitly, e.g. when running via spark-submit " +
              "from an arbitrary working directory."
          )
        }
        dir
    }
  }

  private def loadYaml(file: File): JMap[String, Any] = {
    val yaml = new Yaml()
    val stream = new FileInputStream(file)
    try {
      yaml.load(stream).asInstanceOf[JMap[String, Any]]
    } finally {
      stream.close()
    }
  }

  def loadLabelMapping(root: File): LabelMapping = {
    val raw = loadYaml(new File(root, "configs/label_mapping.json"))
    val mapping = raw.asScala.toMap.map { case (k, v) =>
      k.trim.toLowerCase -> (v match {
        case i: java.lang.Integer => i.intValue()
        case other                => other.toString.toInt
      })
    }
    val ids = mapping.values.toSeq.sorted
    require(
      ids == ids.indices.toSeq,
      s"Label mapping ids must be a contiguous 0..N-1 range, got: $ids"
    )
    LabelMapping(mapping)
  }

  def load(root: File = findProjectRoot()): ProjectConfig = {
    val raw = loadYaml(new File(root, "configs/config.yaml"))
    val labels = loadLabelMapping(root)
    ProjectConfig(root, raw, labels)
  }
}

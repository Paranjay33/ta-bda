ThisBuild / scalaVersion := "2.12.18"
ThisBuild / version      := "0.2.0"
ThisBuild / organization := "edu.project.hinglish"

val sparkVersion = "3.5.1"

// Spark 3.5 on Java 17 needs these module opens (same list spark-submit adds).
lazy val java17Opens = Seq(
  "java.lang", "java.lang.invoke", "java.lang.reflect", "java.io", "java.net", "java.nio",
  "java.util", "java.util.concurrent", "java.util.concurrent.atomic",
  "sun.nio.ch", "sun.nio.cs", "sun.security.action", "sun.util.calendar"
).map(p => s"--add-opens=java.base/$p=ALL-UNNAMED") :+ "-Djdk.reflect.useDirectMethodHandle=false"

lazy val root = (project in file("."))
  .settings(
    name := "hinglish-spark-branch",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % sparkVersion % Provided,
      "org.apache.spark" %% "spark-sql"  % sparkVersion % Provided,
      "org.apache.spark" %% "spark-mllib" % sparkVersion % Provided,
      "org.yaml" % "snakeyaml" % "2.2",
      "org.scalatest" %% "scalatest" % "3.2.18" % Test
    ),
    Test / parallelExecution := false,
    Test / fork := true,
    Test / javaOptions ++= java17Opens ++ Seq("-Xmx2g"),
    // Provided Spark deps must be on the test classpath so `sbt test` works
    // without a separate spark-submit step.
    Test / dependencyClasspath ++= (Compile / dependencyClasspath).value,
    // `sbt run`/`runMain` include Provided deps too (local execution without a
    // Spark distribution). Long-running jobs use scripts/run_spark.sh instead,
    // which launches `java -cp <exported classpath>` directly.
    Compile / run := Defaults.runTask(Compile / fullClasspath, Compile / run / mainClass, Compile / run / runner).evaluated,
    Compile / runMain := Defaults.runMainTask(Compile / fullClasspath, Compile / run / runner).evaluated,
    run / fork := true,
    run / javaOptions ++= java17Opens ++ Seq("-Xmx4g")
  )

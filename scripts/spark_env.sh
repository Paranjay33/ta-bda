#!/usr/bin/env bash
# Shared environment for every Spark/Scala command (sourced, not executed).
#  * JAVA_HOME: Spark 3.5.1 supports Java 8/11/17. Override JAVA_HOME to use
#    your own JDK 17; otherwise the portable Temurin 17 under ../tools is used.
#  * sbt / coursier caches live next to the project (not on a full C: drive).
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export HINGLISH_PROJECT_ROOT="$ROOT_DIR"

if [[ -z "${JAVA_HOME:-}" || ! -x "${JAVA_HOME}/bin/java" && ! -x "${JAVA_HOME}/bin/java.exe" ]]; then
  for cand in "$ROOT_DIR"/../../tools/jdk-17* "$ROOT_DIR"/tools/jdk-17* /usr/lib/jvm/java-17*; do
    if [[ -x "$cand/bin/java" || -x "$cand/bin/java.exe" ]]; then export JAVA_HOME="$(cd "$cand" && pwd)"; break; fi
  done
fi
[[ -n "${JAVA_HOME:-}" ]] && export PATH="$JAVA_HOME/bin:$PATH"

# sbt is always invoked from spark/, and its launcher word-splits SBT_OPTS, so
# cache locations are given RELATIVE to spark/ (the project path contains a space).
export COURSIER_CACHE="$ROOT_DIR/.cache/coursier"
export SBT_OPTS="${SBT_OPTS:-} -Dsbt.global.base=../.cache/sbt-global -Dsbt.boot.directory=../.cache/sbt-boot -Dsbt.ivy.home=../.cache/ivy2 -Dsbt.coursier.home=../.cache/coursier -Xmx2g"
mkdir -p "$ROOT_DIR/.tmp"
export TMP="$ROOT_DIR/.tmp" TEMP="$ROOT_DIR/.tmp"

JAVA17_OPENS=(
  --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED
  --add-opens=java.base/java.lang.reflect=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED
  --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED
  --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED
  --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED
  --add-opens=java.base/sun.nio.cs=ALL-UNNAMED --add-opens=java.base/sun.security.action=ALL-UNNAMED
  --add-opens=java.base/sun.util.calendar=ALL-UNNAMED -Djdk.reflect.useDirectMethodHandle=false
)

# Classpath (compiled classes + Spark/Provided deps), exported once by sbt.
spark_classpath() {
  local cpfile="$ROOT_DIR/spark/target/runtime-classpath.txt"
  if [[ ! -s "$cpfile" || "$ROOT_DIR/spark/build.sbt" -nt "$cpfile" ]]; then
    (cd "$ROOT_DIR/spark" && sbt -batch -no-colors "export Compile / fullClasspath" 2>/dev/null | tail -1 > "$cpfile")
  fi
  cat "$cpfile"
}

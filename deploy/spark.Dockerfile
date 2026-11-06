# Persistent Spark/Scala inference service for E1/E2 (and the BDA jobs).
#   docker build -f deploy/spark.Dockerfile -t hinglishsense-spark .
# Build stage resolves Spark 3.5.1 / Scala 2.12.18 deps with sbt and compiles;
# the runtime stage keeps the classes + resolved jars at the SAME paths so the
# exported classpath stays valid. No Spark distribution or winutils needed.
FROM eclipse-temurin:17.0.13_11-jdk-jammy AS build
ARG SBT_VERSION=1.9.9
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
 && curl -fsSL "https://github.com/sbt/sbt/releases/download/v${SBT_VERSION}/sbt-${SBT_VERSION}.tgz" | tar -xz -C /opt \
 && rm -rf /var/lib/apt/lists/*
ENV PATH=/opt/sbt/bin:$PATH COURSIER_CACHE=/deps/coursier
WORKDIR /app/spark
COPY spark/build.sbt ./build.sbt
COPY spark/project ./project
RUN sbt -batch -Dsbt.boot.directory=/deps/sbt-boot -Dsbt.global.base=/deps/sbt-global update
COPY spark/src ./src
RUN sbt -batch -Dsbt.boot.directory=/deps/sbt-boot -Dsbt.global.base=/deps/sbt-global compile \
 && sbt -batch -no-colors -Dsbt.boot.directory=/deps/sbt-boot -Dsbt.global.base=/deps/sbt-global "export Compile / fullClasspath" | tail -1 > /app/classpath.txt

FROM eclipse-temurin:17.0.13_11-jre-jammy
COPY --from=build /deps /deps
COPY --from=build /app/spark/target /app/spark/target
COPY --from=build /app/classpath.txt /app/classpath.txt
COPY configs /app/configs
RUN useradd --create-home --uid 10001 hs && mkdir -p /app/artifacts /app/data/processed /tmp/spark && chown -R hs /tmp/spark
USER hs
WORKDIR /app
ENV HINGLISH_PROJECT_ROOT=/app SPARK_SERVING_HOST=0.0.0.0 SPARK_SERVING_PORT=8090 SPARK_SERVING_MASTER=local[2] \
    JAVA_OPTS="-Xmx2g --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/sun.nio.cs=ALL-UNNAMED --add-opens=java.base/sun.security.action=ALL-UNNAMED --add-opens=java.base/sun.util.calendar=ALL-UNNAMED -Djdk.reflect.useDirectMethodHandle=false -Djava.io.tmpdir=/tmp/spark"
EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD bash -c 'exec 3<>/dev/tcp/127.0.0.1/8090 && printf "GET /health HTTP/1.0\r\n\r\n" >&3 && grep -q "\"loaded\":true" <&3'
# MAIN_CLASS can be overridden to run E1/E2 training or the scalability benchmark in the same image.
ENV MAIN_CLASS=hinglish.serving.SparkInferenceServer
CMD ["bash", "-c", "exec java $JAVA_OPTS -cp \"$(cat /app/classpath.txt)\" $MAIN_CLASS"]

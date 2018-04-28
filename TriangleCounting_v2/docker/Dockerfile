FROM docker.io/sequenceiq/spark
ADD target /target
WORKDIR /target

ENTRYPOINT ["spark-submit", "--class", "TriangleCountingHW", "tri.jar"]

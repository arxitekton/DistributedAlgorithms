FROM p7hb/docker-spark

ADD artifact /artifact
ADD followers.txt /artifact/src/main/resources/GraphxData/followers.txt

WORKDIR /artifact

ENTRYPOINT ["spark-submit", "--class", "graphxTC.TriangleCountingHW", "trianglecount-1.0-SNAPSHOT.jar"]

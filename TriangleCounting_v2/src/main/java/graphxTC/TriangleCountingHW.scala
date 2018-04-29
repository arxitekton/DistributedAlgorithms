package graphxTC

import org.apache.spark.graphx.{GraphLoader, PartitionStrategy}
import org.apache.spark.sql.SparkSession
import org.apache.spark.graphx.lib.CustomTriCount


object TriangleCountingHW {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("TriangleCountingHW")
      .master("local")
      .getOrCreate()

    /*
    val graph = GraphLoader
      .edgeListFile(spark.sparkContext, "./src/main/resources/GraphxData/higgs-social_network.edgelist", true)
      .partitionBy(PartitionStrategy.RandomVertexCut)
    */

    val graph = GraphLoader
      .edgeListFile(spark.sparkContext, "./src/main/resources/GraphxData/followers.txt", true)
      .partitionBy(PartitionStrategy.RandomVertexCut)

    val tc = CustomTriCount.run(graph).vertices

    println("NUmber of Triangles: " + tc.map(_._2).reduce(_ + _) / 3)

    spark.stop()
  }
}

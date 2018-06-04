package kmeans

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.sql.types._


object DistrKMeans {

  case class Route (
                     id: String,
                     vendor_id: Integer,
                     pickup_datetime: String,
                     dropoff_datetime: String,
                     passenger_count: Integer,
                     pickup_latitude: Double,
                     pickup_longitude: Double,
                     dropoff_latitude: Double,
                     dropoff_longitude: Double,
                     store_and_fwd_flag: String,
                     trip_duration: Double
                   ) extends Serializable

  def main(args: Array[String]): Unit = {

    val spark: SparkSession = SparkSession.builder()
      .appName("DistrKMeans")
      .master("local")
      .getOrCreate()

    import spark.implicits._

    val schema = StructType(Array(
      StructField("id", StringType, true),
      StructField("vendor_id", IntegerType, true),
      StructField("pickup_datetime", TimestampType, true),
      StructField("dropoff_datetime", TimestampType, true),
      StructField("passenger_count", IntegerType, true),
      StructField("pickup_latitude", DoubleType, true),
      StructField("pickup_longitude", DoubleType, true),
      StructField("dropoff_latitude", DoubleType, true),
      StructField("dropoff_longitude", DoubleType, true),
      StructField("store_and_fwd_flag", StringType, true),
      StructField("trip_duration", DoubleType, true)
    ))


    val df: Dataset[Route] = spark.read.option("header", "true").schema(schema).csv("./src/main/resources/kmeans/train.csv.gz").as[Route]

    df.cache
    df.show
    df.schema

    val featureCols = Array("pickup_latitude", "pickup_longitude")
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val df2 = assembler.transform(df)
    val Array(trainingData, testData) = df2.randomSplit(Array(0.9, 0.1))


    val kmeans = new KMeans().setK(7).setFeaturesCol("features").setMaxIter(5)
    val model = kmeans.fit(trainingData)
    println("Final Centers: ")
    model.clusterCenters.foreach(println)

    val categories = model.transform(testData)

    categories.show


    spark.stop()
  }

}

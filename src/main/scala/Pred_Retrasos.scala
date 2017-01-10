import java.io._

import au.com.bytecode.opencsv.CSVReader
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.JavaConverters._


object Pred_Retrasos extends App{

  //Suppress Spark output
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)

  val sparkConfig = new SparkConf()
    .setAppName("PredictDelays")
    .setMaster("local[*]")
    .setSparkHome("$SPARK_HOME")

  // Contexto de spark
  val sparkContext = new SparkContext(sparkConfig)

  /* Leer las variables del dataset vuelos necesarias para evaluar
     Estas variables son las que conocemos antes de que se produzca el vuelo,
     salvo la variable que hay que predecir que es WeatherDelay (0 si no hay retraso y 1 si lo hay)

    Year
    Month
    DayofMonth
    DayOfWeek
    CRSDepTime
    distance
    Cancelled
    WeatherDelay
   */

  //Creamos una clase con el formato del registro que se leerá de los datasets
  case class DelayRec(year: String,
                      month: String,
                      dayOfMonth: String,
                      dayOfWeek: String,
                      crsDepTime: String,
                      origin: String,
                      distance: String,
                      cancelled: String,
                      WeatherDelay: String
                     ) {
    // Creamos el formato de la tupla con las características que serán evaluadas
    def gen_features: (String, Array[Double]) = {
      val rec = Array(
        WeatherDelay.toDouble,
        month.toDouble,
        dayOfMonth.toDouble,
        dayOfWeek.toDouble,
        "%04d".format(crsDepTime.toInt).take(2).toDouble,
        distance.toDouble
      )
      new Tuple2( "%04d%02d%02d".format(year.toInt, month.toInt, dayOfMonth.toInt),rec)
    } }


  def Vuelos200520062007(infile: String): RDD[DelayRec] = {
    val data = sparkContext.textFile(infile)

    data.map { line =>
      val reader = new CSVReader(new StringReader(line))
      reader.readAll().asScala.toList.map(rec => DelayRec (rec(0),rec(1),rec(2),rec(3),rec(5),rec(16),rec(18),rec(21),rec(25)))
    }.map(list => list(0))
      .filter(rec => rec.year != "Year") // Eliminamos la cabecera del fichero
      .filter(rec => rec.cancelled == "0") // Eliminamos los vuelos cancelados
      .filter(rec => rec.WeatherDelay != "NA") // Eliminamos nulos
      .filter(rec => rec.month != "NA") // Eliminamos nulos
      .filter(rec => rec.dayOfMonth != "NA") // Eliminamos nulos
      .filter(rec => rec.crsDepTime != "NA") // Eliminamos nulos
      .filter(rec => rec.distance != "NA") // Eliminamos nulos
  }


  def Vuelos2008(infile: String): RDD[DelayRec] = {
    val data = sparkContext.textFile(infile)

    data.map { line =>
      val reader = new CSVReader(new StringReader(line))
      reader.readAll().asScala.toList.map(rec => DelayRec (rec(0),rec(1),rec(2),rec(3),rec(5),rec(16),rec(18),rec(21),rec(25)))
    }.map(list => list(0))
      .filter(rec => rec.year != "Year") // Eliminamos la cabecera del fichero
      .filter(rec => rec.cancelled == "0") // Eliminamos los vuelos cancelados
      .filter(rec => rec.year == "2008") // Recuperamos solo los vuelos del año 2008 para realizar la evaluación
      .filter(rec => rec.WeatherDelay != "NA") // Eliminamos nulos
      .filter(rec => rec.month != "NA") // Eliminamos nulos
      .filter(rec => rec.dayOfMonth != "NA") // Eliminamos nulos
      .filter(rec => rec.crsDepTime != "NA") // Eliminamos nulos
      .filter(rec => rec.distance != "NA") // Eliminamos nulos
  }

  // Se leen los datos de entrenamiento de los años 2005,2006 y 2007
  val data_200520062007 = Vuelos200520062007("/home/sergio/TFM_Ficheros/Flights/*").map(rec => rec.gen_features._2)
  val data_2008 = Vuelos2008("hdfs://localhost:9000/user/Flights/Flights.csv").map(rec => rec.gen_features._2)

  // Se preparan los datos de entrenamiento
  val parsedTrainData = data_200520062007.map(DefinicionRetraso)
  parsedTrainData.cache
  val scaler = new StandardScaler(withMean = true, withStd = true).fit(parsedTrainData.map(x => x.features))
  val scaledTrainData = parsedTrainData.map(x => LabeledPoint(x.label, scaler.transform(Vectors.dense(x.features.toArray))))
  scaledTrainData.cache



  // Se preparan los datos de validación
  val parsedTestData = data_2008.map(DefinicionRetraso)
  data_2008.map(DefinicionRetraso)
  parsedTestData.cache
  val scaledTestData = parsedTestData.map(x => LabeledPoint(x.label, scaler.transform(Vectors.dense(x.features.toArray))))
  scaledTestData.cache


  // Se contruye el arbol de decision
  val numClasses = 2
  val categoricalFeaturesInfo = Map[Int, Int]()
  val impurity = "gini"
  val maxDepth = 10
  val maxBins = 100
  val model_dt = DecisionTree.trainClassifier(parsedTrainData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)


  //  Se ejecutan las predicciones
  val labelsAndPreds_dt = parsedTestData.map { point =>
    val pred = model_dt.predict(point.features)
    (pred, point.label)
  }
  val m_dt = evaluar_metricas(labelsAndPreds_dt)._2
  println("precision = %.2f, exactitud = %.2f".format(m_dt(0),  m_dt(3)))

  // Evaluamos las métricas obtenidas

  def evaluar_metricas(labelsAndPreds: RDD[(Double, Double)]) : Tuple2[Array[Double], Array[Double]] = {
    val tp = labelsAndPreds.filter(r => r._1==1 && r._2==1).count.toDouble
    val tn = labelsAndPreds.filter(r => r._1==0 && r._2==0).count.toDouble
    val fp = labelsAndPreds.filter(r => r._1==1 && r._2==0).count.toDouble
    val fn = labelsAndPreds.filter(r => r._1==0 && r._2==1).count.toDouble

    val precision = tp / (tp+fp)
    val recall = tp / (tp+fn)
    val F_measure = 2*precision*recall / (precision+recall)
    val exactitud = (tp+tn) / (tp+tn+fp+fn)
    new Tuple2(Array(tp, tn, fp, fn), Array(precision, recall, F_measure, exactitud))

  }

  // Se asume que el retraso se produce cuando el vuelo se retrasa mas de 15 minutos por el clima
  def DefinicionRetraso(vals: Array[Double]): LabeledPoint = {
    LabeledPoint(if (vals(0)>=15) 1.0 else 0.0, Vectors.dense(vals.drop(1)))
  }

  sys.exit(0)


}

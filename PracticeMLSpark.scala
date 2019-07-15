// Databricks notebook source
// Name: Nikhil Kalekar


//  Problem Statement: Machine Learning Practice in Spark. (Simple Data set of Iris)

// COMMAND ----------

val irisDemo = spark.read.option("header","false").option("inferSchema","true").csv("/FileStore/tables/iris.data").toDF("SepalLength","SepalWidth","PetalLength","PetalWidth","Class")

// COMMAND ----------

irisDemo.show(false)

// COMMAND ----------

import org.apache.spark.ml.feature.StringIndexer


// COMMAND ----------

val indexer = new StringIndexer()
.setInputCol("Class").setOutputCol("True-Class")

// COMMAND ----------

val iris = indexer.fit(irisDemo).transform(irisDemo)

// COMMAND ----------

iris.count()

iris.show(150)

// COMMAND ----------

iris.printSchema()

// COMMAND ----------

// convert all the values to double.

val realIris = iris.select("SepalLength","SepalWidth","PetalLength","PetalWidth","True-Class")

// COMMAND ----------

realIris.describe("SepalLength","SepalWidth","PetalLength","PetalWidth").show()

// COMMAND ----------

// lets try with the 3 columns, 1st, 3rd and 4th. as i think SepalWidth is a bit redundant. lets try with 3 and then all columns and compare accuracy. 

// COMMAND ----------

//  vector Assembler... group together all the useful features..
import org.apache.spark.ml.feature.VectorAssembler
val assembler = new VectorAssembler()
.setInputCols(Array("SepalLength","PetalLength","PetalWidth"))
.setOutputCol("FeatureCol")

// COMMAND ----------

val irisFeature = assembler.transform(realIris)

// COMMAND ----------

irisFeature.show()

// COMMAND ----------

// Looking at the data, I think DT would be a good choice.. Some column values are upto certain values like class 1 has some column value till 2.5, next class col values is from 3 to 4.5 and so on...
// lets see...


// COMMAND ----------

import org.apache.spark.ml.classification.DecisionTreeClassifier

// COMMAND ----------

val DT = new DecisionTreeClassifier().setLabelCol("True-Class").setFeaturesCol("FeatureCol")

// COMMAND ----------

// Train Test :

val Array(train, test) = irisFeature.randomSplit(Array(0.75, 0.25))


// COMMAND ----------

val DTModel = DT.fit(train)
val result = DTModel.transform(test)

// COMMAND ----------

result.select("True-Class","prediction").show()

// COMMAND ----------

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// COMMAND ----------

val evaluator = new MulticlassClassificationEvaluator() 
evaluator.setLabelCol("prediction")
evaluator.setMetricName("accuracy")
val accuracy = evaluator.evaluate(result)

// COMMAND ----------



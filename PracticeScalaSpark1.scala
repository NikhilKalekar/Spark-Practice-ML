// Databricks notebook source
// Name: Nikhil Kalekar


// HW1 : 
// Q1) Find the count of all the unique words on the wikipedia :  

// COMMAND ----------

val RawWiki = sc.textFile("/FileStore/tables/text.txt").map(_.toLowerCase()).flatMap(x=>x.split("\\s+"))

// COMMAND ----------

RawWiki.take(10)

// COMMAND ----------

import org.apache.spark.ml.feature.StopWordsRemover

val StopWordsRemoverSet = StopWordsRemover.loadDefaultStopWords("english").toSet



// COMMAND ----------

val wikiWords = RawWiki.filter(x=>x.length>1).filter(x=>StopWordsRemoverSet.contains(x)==false).filter(_.forall(java.lang.Character.isLetter))
wikiWords.take(10)


// ----------------------------------------------------********IMP**********-------------------------------------------------


// VVVVV IMPPPPPPPPPPPPP----- This following code will throw an error as when u do map(_.filter(_.length)) here "_.filter" will be checking for (Example the 1st word in our RDD is "nikhil") nikhil.filter, hence the map will take the array as (n,i,k,h,i,l) and then it will compare _.length with each char, i.e. (n,i,k,h,i,l) so thats why the Error that "Expected:String, Found:Char"
// val demo = RawWiki.map(_.filter(_.length > 3))
// demo.take(3)
// Hence the correct code will be:
// // val demo = RawWiki.filter(x=> x.length > 2)



// COMMAND ----------

// total words (not unique):
wikiWords.count()

// COMMAND ----------

val uniqueWords = wikiWords.map(x=>(x,1)).reduceByKey((x,y)=>x+y)

// COMMAND ----------

// number of unique words in this sample wiki file:
uniqueWords.count

// COMMAND ----------

//  Spark Lab 2 HW:
// PART 1 : UNDERSTANDING THE DATA:

// COMMAND ----------

val movies = sc.textFile("/FileStore/tables/movies.csv")
val ratings = sc.textFile("/FileStore/tables/ratings.csv")
val tags = sc.textFile("/FileStore/tables/tags.csv")

// COMMAND ----------

movies.take(10)
ratings.take(10)

// COMMAND ----------

//  preprocessing for movies :
val MovieHeader = movies.first()
val movie = movies.filter(x=>x!=MovieHeader)
movie.take(10)

// COMMAND ----------

// preprocessing for ratings:
val rHeader = ratings.first()
val rating = ratings.filter(x=>x!=rHeader)
rating.take(10)
// wh have this as multiple strings, so "1,1,4.0,964982703" is one string. 


// COMMAND ----------

// just to check the data in tabluar format..
val movieDf = spark.read.option("header","true").csv("/FileStore/tables/movies.csv")
movieDf.show(false)

// COMMAND ----------

val ratingDf = spark.read.option("header","true").csv("/FileStore/tables/ratings.csv")
ratingDf.show(false)

// COMMAND ----------

// // now since rating RDD has "1,1,4.0,964982703" as one string, we can use map to get multiple arrays of individual strings.. ?  

// COMMAND ----------

rating.take(10)

// COMMAND ----------

val ratingRdd = rating.map(x=>x.split(","))
ratingRdd.take(10)
// column names: userId,movieId,rating,timestamp

// COMMAND ----------

// PART 2: STARTING WITH THE QUESTIONS

// COMMAND ----------

// Q1. Which movie has the highest count of ratings :

// my intuition : get K,V from rating as (MovieId, 1) -> reduceByKey -> join with Movie RDD -> sort according to number of ratings -> get the name.

// getting K,V
val ratingKV = ratingRdd.map(x => (x(1),1)).reduceByKey((x,y)=>x+y)
ratingKV.take(10)

// COMMAND ----------

ratingKV.sortBy(-_._2).take(10)

// COMMAND ----------

// join to get movie name, to do this we need movies RDD, and MoviesRDDJoin(MovieId, Name) :
// movie lables : movieId|title |genres 
val movieRdd = movie.map(x=>x.split(","))
val movieJoinRDD = movieRdd.map(x=>(x(0),x(1)))
movieJoinRDD.take(10)

// COMMAND ----------

//  join RDD based on movieId:

val result = ratingKV.join(movieJoinRDD)
result.sortBy(-_._2._1).take(10)

// COMMAND ----------

// Q2 Find the movie with the lowest count of ratings

result.sortBy(_._2._1).take(10)


// COMMAND ----------

// Q3 Find the avg ratings for each movie:

// intuition : get K,v as id, rating -> get avg ????
// column names: userId,movieId,rating,timestamp

// COMMAND ----------

// convert Rating to float
val ratingIdRat = ratingRdd.map(x=>(x(1),x(2).toFloat))
ratingIdRat.take(10)

// COMMAND ----------

//  calculating the avg... Demo:

val res1 = ratingIdRat.mapValues(x=>(x,1))
//  get K,V as (Id, (Rating, 1))
res1.take(10)

//  reduce by key which has multiple array values: we usually do ReduceByKey((x,y)=>x+y) , here there is a single value for 1 key, hence 1 x and for other/similar key 1 y

val res2 = res1.reduceByKey
{
  case ((rat,one),(rat1,one1)) => ((rat+rat1),(one+one1))
}

res2.take(10)

val result = res2.mapValues{
  case (rat,count) => rat/count
}

result.collect()

// COMMAND ----------

// Q4 Find the movies with the highest avg rating
//  this gives the wrong movie ratings :(  as only 1 person could have rated a bad movie as 5/5. but this is the right average ratings. 
result.sortBy(-_._2).take(20)

// COMMAND ----------

// Q5 Find the movies with the lowest avg ratings

result.sortBy(_._2).take(20)

// COMMAND ----------

// 6. Now join the movies and ratings tables, and give the names of the top 10 movies with the highest ratings
// Hint: use join function
// Already done on cmd 21 and 22

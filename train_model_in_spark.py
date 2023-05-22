import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.ml.feature import Binarizer, Bucketizer, OneHotEncoder, VectorAssembler, StringIndexer, MinMaxScalerModel, \
StandardScaler, Imputer, Tokenizer,StopWordsRemover, MinMaxScaler, PolynomialExpansion
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator, ClusteringEvaluator, RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics

import numpy as np; import pandas as pd
from pyspark.sql.functions import concat, lit

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark import SparkConf

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.linalg import Vectors

if __name__ == "__main__":
    spark = SparkSession.builder.appName("model").config("spark.master", "spark://mn1.bigdata.com:7077").config("spark.hadoop.dfs.client.use.datanode.hostname", "true").getOrCreate()

    #spark.conf.set('spark.sql.adaptive.enabled','false')
    spark.sparkContext.setLogLevel("INFO")

    ###### Read Parquet File
    print("Read Input")

    features = np.load('/home/spark/image_array_86.npy')[::2]
    names = np.load('/home/spark/image_name_86.npy')[::2]
    
    ###### Data Transform
    print("Data Transform")
       
    names = [int(x) for x in names]
    features = [Vectors.dense(features[i]) for i in range(len(features))]
    schema = StructType([StructField("names", IntegerType()),StructField("features", VectorUDT(), True)])
    
    data = [(names[i], features[i]) for i in range(len(names))]
    df2 = spark.createDataFrame(data, schema)
    df2 = df2.withColumn('names', concat(lit('0'), df2['names']))
    
    df3 = df.join(df2,df.article_id ==  df2.names,"inner")
    df3 = df3.select(['article_id','labels','label','features'])
    df3.show(5)
    
    #### Modelling
    print('Modelling')
    
    model_df=df3.select(['features','label'])
    train, test = model_df.randomSplit([0.7,0.3], 42)
    
    
    print('Fit Logistic')
    model = LogisticRegression(maxIter=10, regParam=0.05, elasticNetParam=0.3,featuresCol='features', labelCol="label")
    p_model = model.fit(train)
    df_test_LR = p_model.transform(test)
    
    print('Evaluation')
    evaluator_LR = MulticlassClassificationEvaluator(predictionCol="prediction")
    print("LOGISTIC REGRESSION:")
    print("accuracy: " , evaluator_LR.evaluate(df_test_LR, {evaluator_LR.metricName: "accuracy"}))
    print("precision: " ,evaluator_LR.evaluate(df_test_LR, {evaluator_LR.metricName: "weightedPrecision"}))
    print("recall: " ,evaluator_LR.evaluate(df_test_LR, {evaluator_LR.metricName: "weightedRecall"}))
    print("f1: " ,evaluator_LR.evaluate(df_test_LR, {evaluator_LR.metricName: "f1"}))
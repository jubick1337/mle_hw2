import logging
import os
import shutil
import configparser

from pyspark import SparkConf, SparkContext
from pyspark.ml.feature import HashingTF, IDF
from pyspark.mllib.linalg.distributed import MatrixEntry, CoordinateMatrix
from pyspark.sql import SparkSession

logging.basicConfig(level=logging.INFO)


class FeaturesExtractor:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(os.getcwd(), 'config.ini'))
        # Initialize Spark stuff
        self.spark_config = SparkConf()
        self.spark_config.set("spark.app.name", "hw2")
        self.spark_config.set("spark.master", "local")
        self.spark_config.set("spark.executor.cores",
                              self.config.get("SPARK", "NUM_PROCESSORS"))
        self.spark_config.set("spark.executor.instances",
                              self.config.get("SPARK", "NUM_EXECUTORS"))
        self.spark_config.set("spark.executor.memory", "16g")
        self.spark_config.set("spark.locality.wait", "0")
        self.spark_config.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        self.spark_config.set("spark.kryoserializer.buffer.max", "2000")
        self.spark_config.set("spark.executor.heartbeatInterval", "6000s")
        self.spark_config.set("spark.network.timeout", "10000000s")
        self.spark_config.set("spark.shuffle.spill", "true")
        self.spark_config.set("spark.driver.memory", "16g")
        self.spark_config.set("spark.driver.maxResultSize", "16g")
        self.spark_config.set("spark.sql.parquet.compression.codec", "gzip")

        self.num_parts = int(self.config.get("SPARK", "NUM_PARTS"))

        self.sc = SparkContext.getOrCreate(conf=self.spark_config)
        self.spark_session = SparkSession(self.sc)

        logging.info("Features Extractor initialized")

    def extract(self):
        logging.info(f'Computing features for {self.config.get("DATA", "TRAIN_FILE")}')

        logging.info('Getting matrix of views')

        # Group movies by user ID
        group_by_user_id = self.sc.textFile(self.config.get("DATA", "TRAIN_FILE")).map(
            lambda x: map(int, x.split())).groupByKey().map(lambda x: (x[0], list(x[1])))

        # Convert grouped data to matrix
        matrix = CoordinateMatrix(
            group_by_user_id.flatMapValues(lambda x: x).map(lambda x: MatrixEntry(x[0], x[1], 1.0)))

        shutil.rmtree(self.config.get('DATA', 'MATRIX_PATH'), ignore_errors=True)
        matrix.entries.toDF().write.parquet(self.config.get('DATA', 'MATRIX_PATH'))

        logging.info('Calculating TF')
        tf = HashingTF(inputCol="movie_id", outputCol="tf_features",
                       numFeatures=int(self.config.get("MODELS_PARAMS", "TF_FEATURES_COUNT")))
        tf_features = tf.transform(group_by_user_id.toDF(schema=["user_id", "movie_id"]))
        shutil.rmtree(self.config.get('DATA', 'TF_FEATURES_PATH'), ignore_errors=True)
        tf_features.write.format("parquet").save(self.config.get('DATA', 'TF_FEATURES_PATH'))

        logging.info('Calculating IDF')
        idf = IDF(inputCol="tf_features", outputCol="idf_features")
        idf = idf.fit(tf_features)
        shutil.rmtree(self.config.get('DATA', 'IDF_PATH'), ignore_errors=True)
        idf.write().save(self.config.get('DATA', 'IDF_PATH'))
        idf_features = idf.transform(tf_features)
        shutil.rmtree(self.config.get('DATA', 'IDF_FEATURES_PATH'), ignore_errors=True)
        idf_features.write.format("parquet").save(self.config.get('DATA', 'IDF_FEATURES_PATH'))


if __name__ == "__main__":
    feature_extractor = FeaturesExtractor()
    feature_extractor.extract()

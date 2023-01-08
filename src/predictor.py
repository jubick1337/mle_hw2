import argparse
import logging
import os
import configparser
import random

from pyspark import SparkContext, SparkConf
from pyspark.ml.feature import IDFModel
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
from pyspark.mllib.linalg.distributed import MatrixEntry, CoordinateMatrix

from pyspark.sql import SparkSession

logging.basicConfig(level=logging.INFO)


class Predictor:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(os.getcwd(), 'config.ini'))
        self.spark_config = SparkConf()
        self.spark_config.set("spark.app.name", "homework")
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

        self.idf_features = self.spark_session.read.load(self.config.get('DATA', 'IDF_FEATURES_PATH'))
        self.idf = IDFModel.load(self.config.get('DATA', 'IDF_PATH'))
        self.tf_features = self.spark_session.read.load(self.config.get('DATA', 'TF_FEATURES_PATH'))
        self.matrix = CoordinateMatrix(
            self.spark_session.read.parquet(self.config.get('DATA', 'MATRIX_PATH')).
            rdd.map(lambda x: MatrixEntry(*x)))
        logging.info('Predictor initialized')

    def predict_random(self, top_k):
        logging.info('Generating recommendation for random user')

        # I can't process the whole dataset, so I only used this user for test
        user_id = 276480  # random.randint(0, self.matrix.numCols())
        logging.info(f'Random user is {user_id}')
        # Calculate similarity of users and filter by user_id
        filtered_ids = IndexedRowMatrix(
            self.idf_features.rdd.map(lambda x: IndexedRow(x["user_id"], Vectors.dense(x["idf_features"])))). \
            toBlockMatrix().transpose().toIndexedRowMatrix().columnSimilarities().entries. \
            filter(lambda x: x.i == user_id or x.j == user_id)

        sorted_similarities = IndexedRowMatrix(
            filtered_ids.sortBy(lambda x: x.value, ascending=False
                                ).map(lambda x: IndexedRow(x.j if x.i == user_id else x.i, Vectors.dense(x.value))))

        recommendations = sorted_similarities.toBlockMatrix().transpose().multiply(
            self.matrix.toBlockMatrix()).transpose().toIndexedRowMatrix().rows.sortBy(lambda x: x.vector.values[0],
                                                                                      ascending=False)
        logging.info(
            f'For user {user_id} top {top_k} recommended are: {[x.index for x in recommendations.collect()[:top_k]]}'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()
    predictor = Predictor()
    predictor.predict_random(args.top_k)

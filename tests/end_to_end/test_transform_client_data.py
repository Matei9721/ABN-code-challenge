import unittest
from pyspark.sql import SparkSession
from src.data_processing import transform_data
from chispa.dataframe_comparer import assert_df_equality


class ClientDataProcessingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize SparkSession
        cls.spark = SparkSession.builder.appName("ABN-assignment").getOrCreate()

    @classmethod
    def tearDownClass(cls):
        # Stop SparkSession
        cls.spark.stop()

    def test_e2e_data_processing(self):
        processed_data = transform_data(
            "data/input/dataset_one.csv",
            "data/input/dataset_two.csv",
            ["Netherlands", "United Kingdom"],
        )

        expected_data = self.spark.read.csv(
            "client_data/transformed_client_data.csv", header=True
        )

        assert_df_equality(processed_data, expected_data)

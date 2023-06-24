import unittest
from pyspark.sql import SparkSession
from src.transformers import CountryFilter, ColumnRemover, RenameColumns
from chispa.dataframe_comparer import assert_df_equality


class TransformersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize SparkSession
        cls.spark = SparkSession.builder.appName("ABN-assignment").getOrCreate()

    @classmethod
    def tearDownClass(cls):
        # Stop SparkSession
        cls.spark.stop()

    def test_rename_columns(self):
        # Create sample DataFrame
        data = [(1, "John"), (2, "Alice"), (3, "Bob")]
        columns = ["id", "name"]
        df = self.spark.createDataFrame(data, columns)

        # Define the transformer
        transformer = RenameColumns(
            column_dict={"id": "customer_id", "name": "customer_name"}
        )

        # Apply the transformation
        transformed_df = transformer.transform(df)

        # Check the renamed columns
        expected_columns = ["customer_id", "customer_name"]
        assert transformed_df.columns == expected_columns

    def test_filter_countries(self):
        # Create sample DataFrame
        data = [("John", "USA"), ("Alice", "UK"), ("Bob", "Canada")]
        columns = ["name", "country"]
        df = self.spark.createDataFrame(data, columns)

        # Define the transformer
        transformer = CountryFilter(countries_to_filter=["USA", "UK"])

        # Apply the transformation
        transformed_df = transformer.transform(df)

        # Check the filtered rows
        expected_data = [("John", "USA"), ("Alice", "UK")]
        expected_df = self.spark.createDataFrame(expected_data, columns)

        assert_df_equality(transformed_df, expected_df)

    def test_remove_columns(self):
        # Create sample DataFrame
        data = [(1, "John", 25), (2, "Alice", 30), (3, "Bob", 35)]
        columns = ["id", "name", "age"]
        df = self.spark.createDataFrame(data, columns)

        # Define the transformer
        transformer = ColumnRemover(columns_to_remove=["name", "age"])

        # Apply the transformation
        transformed_df = transformer.transform(df)

        # Check the remaining columns
        expected_columns = ["id"]
        self.assertEqual(transformed_df.columns, expected_columns)

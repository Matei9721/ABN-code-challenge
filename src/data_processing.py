from src.transformers import ColumnRemover, CountryFilter, RenameColumns
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from src.custom_logging import SingletonLogger

logger = SingletonLogger()
column_name_mappings = {
    "id": "client_identifier",
    "btc_a": "bitcoin_address",
    "cc_t": "credit_card_type",
}


def transform_data(
    dataset1: str = None, dataset2: str = None, countries_to_filter: list = []
):
    logger.debug(f"Dataset 1: {dataset1}")
    logger.debug(f"Dataset 2: {dataset2}")
    logger.debug(f"Countries to filter {countries_to_filter}")

    # Initialize spark
    spark = SparkSession.builder.appName("ABN-assignment").getOrCreate()

    # Read datasets
    clients = spark.read.csv(dataset1, header=True)
    client_data = spark.read.csv(dataset2, header=True)

    logger.info("Finished reading data")

    # Initialize transformers
    country_filter = CountryFilter(
        countries_to_filter=["Netherlands", "United Kingdom"]
    )
    remove_pii_clients = ColumnRemover(columns_to_remove=["first_name", "last_name"])
    column_renamer = RenameColumns(column_dict=column_name_mappings)
    remove_credit_card = ColumnRemover(columns_to_remove=["cc_n"])

    # Create pipelines
    client_pipeline = Pipeline(stages=[remove_pii_clients, country_filter])
    client_data_pipeline = Pipeline(stages=[remove_credit_card])

    # Fit, transform
    client_fit = client_pipeline.fit(clients)
    client_output = client_fit.transform(clients)
    client_data_fit = client_data_pipeline.fit(client_data)
    client_data_output = client_data_fit.transform(client_data)

    logger.info("Finished processing data")

    # Join datasets
    final_data = client_output.join(client_data_output, on="id")

    # Rename columns
    final_data = column_renamer.transform(final_data)

    logger.info("Datasets joined and filtered successfully.")

    return final_data

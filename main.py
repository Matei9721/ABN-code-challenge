import argparse
from src.data_processing import transform_data
from src.custom_logging import SingletonLogger


def main():
    """
    Driver function for the script that reads the terminal arguments
    and passes them to the transformation function.
    :return: Void
    """
    # Create logger
    logger = SingletonLogger()

    # Argument parser (Can be done in a separate class if the code grows)
    parser = argparse.ArgumentParser(
        description="KommatiPara Script for retrieving client data."
    )

    # Add arguments with descriptions
    parser.add_argument("-d1", "--dataset1", type=str, help="Path to first dataset")
    parser.add_argument("-d2", "--dataset2", type=str, help="Path to second dataset")
    parser.add_argument(
        "-c",
        "--countries_to_filter",
        default=[],
        nargs="+",
        help="Countries to filter",
        type=str,
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    logger.info("Arguments parsed successfully, starting processing steps!")

    # retrieve data
    final_data = transform_data(args.dataset1, args.dataset2, args.countries_to_filter)

    logger.info("Data processing finished successfully, saving output to disk.")
    # Export data
    final_data.toPandas().to_csv(
        "../client_data/transformed_client_data.csv", index=False
    )


if __name__ == "__main__":
    # Call the main function
    main()

import argparse
from data_processing import transform_data


def main():
    # Argument parser
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

    # retrieve data
    final_data = transform_data(args.dataset1, args.dataset2, args.countries_to_filter)

    # Export data
    final_data.toPandas().to_csv(
        "../client_data/transformed_client_data.csv", index=False
    )


if __name__ == "__main__":
    # Call the main function
    main()

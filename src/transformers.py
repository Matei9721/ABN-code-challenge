from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasOutputCols, Param, Params
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
import pyspark.sql.functions as F
from pyspark.sql import DataFrame


class RenameColumns(
    Transformer,
    HasOutputCols,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    column_dict = Param(
        Params._dummy(),
        "column_dict",
        "Dictionary with new names for the column to be renamed.",
    )

    @keyword_only
    def __init__(self, column_dict: dict = {}):
        """
        Constructor for the transformer. Sets the default parameters.
        :param column_dict: Python dictionary that contains
         old_column_name: new_column_name
        """
        super().__init__()
        self._setDefault(column_dict=column_dict)
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def getColumnDict(self):
        """
        Gets the value of column_dict or its default value.
        """
        return self.getOrDefault(self.column_dict)

    def _transform(self, dataset: DataFrame):
        """
        Applies this transformer's transformation on the dataset.

        :param dataset: PySpark DataFrame to be transformed.
        :return: Transformed PySpark DataFrame.
        """
        column_mapping = self.getColumnDict()
        all_columns = dataset.columns
        # Get list of column that need to be renamed
        renamed_columns = [
            F.col(column).alias(column_mapping[column])
            if column in column_mapping
            else F.col(column)
            for column in all_columns
        ]
        return dataset.select(renamed_columns)


class CountryFilter(
    Transformer,
    HasOutputCols,
    DefaultParamsReadable,
    DefaultParamsWritable,
):

    countries_to_filter = Param(
        Params._dummy(),
        "countries_to_filter",
        "What countries to keep in the filtered dataframe.",
    )

    @keyword_only
    def __init__(self, countries_to_filter: list = ["Netherlands", "United Kingdom"]):
        """
        Constructor for the transformer. Sets the default parameters.

        :param countries_to_filter: Python List of countries to filter.
        """
        super().__init__()
        self._setDefault(countries_to_filter=countries_to_filter)
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def getCountriesToFilter(self):
        """
        Gets the value of countries_to_filter or its default value.
        """
        return self.getOrDefault(self.countries_to_filter)

    def _transform(self, dataset: DataFrame):
        # Avoid throwing errors if column does not exist.
        if "country" in dataset.columns:
            return dataset.filter(F.col("country").isin(self.getCountriesToFilter()))
        else:
            return dataset


class ColumnRemover(
    Transformer,
    HasOutputCols,
    DefaultParamsReadable,
    DefaultParamsWritable,
):

    columns_to_remove = Param(
        Params._dummy(),
        "columns_to_remove",
        "Column to remove",
    )

    @keyword_only
    def __init__(self, columns_to_remove: list = ["first_name", "last_name"]):
        """
        Constructor for the transformer. Sets the default parameters.

        :param columns_to_remove: Python List of columns to be removed.
        """
        super().__init__()
        self._setDefault(columns_to_remove=columns_to_remove)
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def getColumnToRemove(self):
        """
        Gets the value of columns_to_remove or its default value.
        """
        return self.getOrDefault(self.columns_to_remove)

    def _transform(self, dataset: DataFrame):
        """
        Applies this transformer's transformation on the dataset.
        :param dataset: PySpark DataFrame to be transformed
        :return: Pyspark DataFrame
        """
        return dataset.drop(*self.getColumnToRemove())

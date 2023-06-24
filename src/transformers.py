from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasOutputCols, Param, Params
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
import pyspark.sql.functions as F


class RenameColumns(
    Transformer,
    HasOutputCols,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    column_dict = Param(
        Params._dummy(),
        "column_dict",
        "What countries to keep.",
    )

    @keyword_only
    def __init__(self, column_dict={}):
        super().__init__()
        self._setDefault(column_dict=["first_name", "last_name"])
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def getColumnDict(self):
        """
        Gets the value of :py:attr:`value` or its default value.
        """
        return self.getOrDefault(self.column_dict)

    def _transform(self, dataset):
        column_mapping = self.getColumnDict()
        all_columns = dataset.columns
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
        "What countries to keep.",
    )

    @keyword_only
    def __init__(self, countries_to_filter=["first_name", "last_name"]):
        super().__init__()
        self._setDefault(countries_to_filter=["first_name", "last_name"])
        kwargs = self._input_kwargs
        self._set(**kwargs)

    @keyword_only
    def setParams(self, countries_to_filter=["first_name", "last_name"]):
        """
        setParams(self, value=0.0)
        Sets params for this SetValueTransformer.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setValue(self, countries_to_filter):
        """
        Sets the value of :py:attr:`value`.
        """
        return self._set(countries_to_filter=countries_to_filter)

    def getCountriesToFilter(self):
        """
        Gets the value of :py:attr:`value` or its default value.
        """
        return self.getOrDefault(self.countries_to_filter)

    def _transform(self, dataset):
        return dataset.filter(F.col("country").isin(self.getCountriesToFilter()))


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
    def __init__(self, columns_to_remove=["first_name", "last_name"]):
        super().__init__()
        self._setDefault(columns_to_remove=["first_name", "last_name"])
        kwargs = self._input_kwargs
        self._set(**kwargs)

    @keyword_only
    def setParams(self, columns_to_remove=["first_name", "last_name"]):
        """
        setParams(self, value=0.0)
        Sets params for this SetValueTransformer.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setValue(self, columns_to_remove):
        """
        Sets the value of :py:attr:`value`.
        """
        return self._set(columns_to_remove=columns_to_remove)

    def getColumnToRemove(self):
        """
        Gets the value of :py:attr:`value` or its default value.
        """
        return self.getOrDefault(self.columns_to_remove)

    def _transform(self, dataset):
        return dataset.drop(*self.getColumnToRemove())

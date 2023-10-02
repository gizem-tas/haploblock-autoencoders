"""recodeA - a module to handle plink --recode A .raw file data
===========================================================
This module contains methods to import, summarise, impute
and transform .raw files with recoded data. This type of file
is obtained with

`plink --bfile <bfile_name> --recode A <output_file_name>`

and has the following format:

FID  IID  PAT  MAT  SEX  PHENOTYPE snp001  snp002 ...  snpXXX

Where each snp value of every sample is a 0, 1 or 2, for zero,
one or two minor variant alleles.
"""

import pandas as pd
import numpy as np


class RecodedData(pd.DataFrame):
    """Data class for .raw files obtained using `plink --recode A`
    .raw files can either be read using `recodeA.read_raw()` to
    transform them into a RecodedData object, or a pandas.DataFrame
    with .raw data can be converted into a RecodeData.
    
    Arguments:
        pd.DataFrame
    """
    
    @property
    def _constructor(self):
        """Turns pandas.DataFrame into a RecodedData object.
        
        Returns:
            RecodedData object
        """
        return RecodedData

    def summarise(self):
        """Prints a summary of .raw data.
        """
        
        sexes = self.SEX.value_counts()
        pheno = self.PHENOTYPE.value_counts()
        total = self.shape[0]
        print(f"{self.shape[1]-6} variants.\n{self.shape[0]} people ({sexes.loc[1]} males, {sexes.loc[2]} females, {total - sexes.loc[1] - sexes.loc[2]} ambiguous).\n{pheno.loc[1] + pheno.loc[2]} phenotype values.\n{pheno.loc[2]} cases, {pheno.loc[1]} controls, {total - pheno.loc[1] - pheno.loc[2]} missing.\n{self.isna().sum().sum()} missing values.")

    def span(self, end = None):
        """Returns the span of the .raw snp data in the 
        format <chr>:<start_snp>:<end_snp>. It is optional 
        to provide a string character append to the span.
        
        Arguments:
            end: character to append (str), default None
        
        Returns:
            span str
        """
        
        chrom, start_snp = self.columns[6].split(sep=":")[:2]
        end_snp  = self.columns[-1].split(sep=":")[1]
        snp_span = ":".join([chrom, start_snp, end_snp])

        return snp_span + end if end else snp_span

    def impute(self, imputation = "mode"):
        """Imputs the variants data. Na's are replaced with 
        either the mode of the columns (`mode` setting) or 
        with zeros (`zero` setting`)
        
        Arguments:
            imputation: type of imputation (str: `mode`|`zero`), default `mode`
            
        Returns:
            Imputed RecodeData object
        """
        
        if imputation == "zero":
            fill = 0
        elif imputation == "mode":
            fill = self.iloc[:,6:].mode().iloc[0]
        else:
            raise ValueError(f"Not a valid imputation type ({imputation}), must be either \"zero\" or \"mode\"")

        return pd.concat([self.iloc[:,:6], self.iloc[:,6:].fillna(fill)], axis=1)

    def import_split(self, train_file, test_file):
        """Splits samples into premade train and test split,
        based on the IID columns in the train and test file.
        
        Arguments:
            train_file: pandas.Dataframe or file path of train file (pandas.DataFrame|str)
            test_file: pandas.Dataframe or file path of test file (pandas.DataFrame|str)
            
        Returns:
            tuple(RecodeData train data, RecodeData test data)
        """
        
        if isinstance(train_file, str):
            train_file = pd.read_table(train_file, header=None)
        if isinstance(test_file, str):
            test_file  = pd.read_table(test_file, header=None)
        
        train_indices = train_file.iloc[:,0].values.tolist()
        test_indices = test_file.iloc[:,0].values.tolist()
        
        train_data = self.loc[self["IID"].isin(train_indices)].reset_index(drop=True)
        test_data  = self.loc[self["IID"].isin(test_indices)].reset_index(drop=True)

        return train_data, test_data

    def get_variants(self):
        """Extracts the variants from the RecodedData object and
        returns it as a numpy array.
        
        Returns:
            numpy.ndarray variants data
        """
        if self.isna().values.any():
            raise ValueError("Data contains NaN values, use RecodedData.impute() to impute missing values.")

        return np.asarray(self.iloc[:,6:].astype("int8"))

    def get_info(self):
        """Extracts the sample inforamtion from the RecodeData
        object and returns it as a RecodedInfo object.
        
        Returns:
            RecodedInfo object
        """
        return RecodedInfo(self.iloc[:,:6])

    def save(self, file_path):
        self.to_csv(file_path, sep=" ", index=False)


class RecodedInfo(pd.DataFrame):
    """Data class for the sample information of 
    .raw files. The class is created using the
    `RecodedData.get_info()` method.
    """
    
    @property
    def _constructor(self):
        """Turns pandas.DataFrame-like object into 
        a RecodedInfo object.
        
        Returns:
            RecodedInfo object
        """
        return RecodedInfo

    def join_variants(self, variants, col_names = "FEAT_"):
        """Joins variants data t'o the sample information
        and returns it as a RecodedData object. The
        `RecodedData.span(end=":)` method can be used to 
        set columns names.
        """
        
        if self.shape[0]!=variants.shape[0]:
            raise ValueError(f"Mismatch in number of samples ({self.shape[0]} rows vs {variants.shape[0]} rows)")

        joined_data = pd.concat([self.reset_index(drop=True), pd.DataFrame(variants)], axis=1)
        joined_data.columns = list(self.columns) + [col_names + str(i) for i in range(1, variants.shape[1]+1)]

        return RecodedData(joined_data)


def read_raw(raw_file_path: str):
    """Function for reading a .raw file into a
    RecodedData object.
    
    Arguments:
        raw_file_path: file path of .raw file (str)
    
    Returns:
        RecodedData object
    """
    
    file = pd.read_csv(raw_file_path, sep=" ")
    if list(file.columns[:6]) == ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']:
        return RecodedData(file)
    else:
        raise ValueError("File is not a .raw recoded data file")

This directory contains the python scripts that are used to handle and compress the recoded data.

## recodeA.py:

A module for importing, summarizing, imputing and transforming .raw file with recoded data. A recoded data file is obtained with
```bash
plink --bfile <bfile_name> --recode A <output_file_name>`
```

and has the following format:
```bash
FID  IID  PAT  MAT  SEX  PHENOTYPE snp001  snp002 ...  snpXXX
```
Where each snp value of every sample is a 0, 1 or 2, for zero,
one or two minor variant alleles.

.raw data is imported with `recodeA.read_raw(<raw_file>)` and returns a RecodedData object (inherits from pandas.DataFrame). The RecodedData class methods are documented in the file itself. For use in a autoencoder, the data must be transformed into a 2D numpy array. This is done with the `recodeA.RecodedData.get_variants()` method. The sample information can be extracted using `recodeA.RecodedData.get_info()` and returns a RecodedInfo object. The sample information can be joined with a numpy array containing variants data (must have the same number of samples).

### Examples:
---------

extracting the variants from a .raw file
```python
import recodeA
data = recodeA.read_raw("path_to_raw_file")
data_imputed = raw.impute(mode="zero")
variants = data_imputed.get_variants()
print(type(data), type(variants))
<class 'RecodedData'>, <class 'numpy.ndarray'>
```

## autoencoder.py:

A module for creating and wrapping autoencoder models using the tensorflow API.

The module contains functions to create a tensorflow.keras.Sequential autoencoder moedel, as well as a wrapper API class to use the autoencoder model with other libraries and API's, like the sklearn.model_selection.GridSearchCV API.

The main function of the module is `autoencoder.create_model()`. This function creates a tensorflow.keras.Sequential autoencoder model that can be trained with 2D data. The shape of the data should by (n_samples, n_inputs). The input data should be the same as the output data, since autoencoders are trained to attempt a reconstruction of this data.

The shape of the autoencoder, hidden layer number and bottleneck size are parameters of the function. The model is optimized with the Adam optimizer, the loss is MSE and used a custom metric called "snp_accuracy". The activation funciton of the hidden layers is LeakyReLU, the acivation of the output layers is a custom function called "additive", which has an output range from 0 to 2.

To use a tensorflow model in a gridsearch API such as sklearn.model_selection.GridSearchCV, the model has to be wrapped in a Wrapper API. The standard tensorflow.keras.wrappers.scikit_learn wrapper does not work with models with a custom accuracy. Therefore, I reimplemented the API to be compatible with the autoencoder models. The new wrapper class, called Classifier, has the same functions and behaviour as the original API, but is optimized for the autoencoder model. To wrap an autoencoder, first build the model like `model = autoencoder.create_model(inputs=100, shape="sqrt", hl=3, bn=3)`. Then wrap the model with the Classifier class: `estimator = autoencoder.Classifier(build_fn=model)`. The "scoring" parameter defines how the 'best' model is selected. For autoencoders, this can either be "snp_accuracy" or "loss". Accuracy optimization seeks a maximum, while loss optimization seeks a minimum. If the latter is being optimized, the "greater_is_better" parameter should be set to "True". This will return negative loss values, ensuring the optimization selects the lowest absolute loss.

After a tensorflow.keras.Sequential autoencoder model is sufficiently trained, the encoder can be extracted to predict the bottleneck values. The function `extract_encoder()` return the encoder part of an autoencoder as a tensorflow.keras.sequential model. With the `.predict()` function, the bottleneck values are calculated.


### Examples:
---------

predicting autoencoder bottleneck values

```python
import autoencoder as ae
import recodeA
data = recodeA.read_raw("path_to_raw_file").impute().get_variants() # using .raw data
model = ae.create_model(inputs=data.shape[1], shape="sqrt:, hl=3, bn=3)
history = model.fit(data, data)
encoder = ae.extract_encoder(model)
encoder.predict(data)
array([[ 3.0684285 ,  1.2678118 ,  0.5405439 ],
       [ 2.7718604 , 23.414883  ,  4.911639  ],
       ...,
       [11.091528  , 14.339089  ,  5.6872787 ],
       [ 0.6532019 ,  4.692735  ,  4.095155  ]], dtype=float32)
```

using an autoencoder model in a GridSearchCV

```python
import autoencoder as ae
import recodeA
from sklearn.model_selection import GridSearchCV
data = recodeA.read_raw("path_to_raw_file").impute().get_variants() # using .raw data
model = ae.create_model(inputs=data.shape[1], shape="sqrt:, hl=3, bn=3)
```

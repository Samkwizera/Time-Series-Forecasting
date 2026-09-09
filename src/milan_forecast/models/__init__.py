"""Forecasting models. Every module exposes ``forecast(data, part, params, ...)`` returning a
tidy prediction table (see ``datasets.predictions_frame``) and, for learned models, an info dict.

Common protocol
---------------
* ``part="val"``: fit on the training split, forecast every origin whose targets fall in the
  validation split.
* ``part="test"``: fit on train + validation, forecast every origin whose targets fall in the
  test split. The test split is therefore touched exactly once per final configuration.
* Early stopping never looks at the evaluation split: the last week of the fitting window is
  held out for that purpose and the model is refit on the whole window afterwards.
"""

from . import baseline, lgbm, rnn, sarima  # noqa: F401

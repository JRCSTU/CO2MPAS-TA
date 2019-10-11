###############
|co2mpas| Model
###############

.. module:: co2mpas.core.model
   :noindex:

CO2MPAS model is plotted here below: you can explore the diagram nests by
clicking on them.

.. _model_diagram:

.. dispatcher:: dsp
   :opt: depth=1, index=True
   :alt: Flow-diagram of the execution of various Stages and Cycles sub-models.
   :width: 640

  >>> from co2mpas.core.model import dsp
  >>> import schedula
  >>> dsp = dsp.register(memo={})

The execution of |co2mpas| model for a single vehicle is a procedure in three
sequential stages:

  - **Calibration stage**: identifies, calibrates, and selects the best
    physical models (see next section `Model selection`_) from WLTP input data
    (i.e., ``input.calibration.<cycle>``).
  - **Model selection stage**: selects the best calibrated models
    (i.e., ``data.prediction.models_<cycle>``) to be used in the prediction stage.
  - **Prediction stage**: forecasts the |CO2| emissions using the user's inputs
    (i.e., ``input.prediction.<cycle>``) and the calibrated models. If some/all
    WLTP inputs are not provided, the function
    :py:func:`select_prediction_data <select_prediction_data>` chooses those
    required to predict |CO2| emissions from ``output.calibration.<cycle>``.


.. module:: co2mpas.core.model.selector.models
   :noindex:

The :py:obj:`physical model <co2mpas.core.model.physical.dsp>` is used in both
stages: calibration (i.e., ``calibrate_with_<cycle>``) and prediction (i.e.,
``predict_<cycle>``). The identified/calibrated parameters from WLTP
data (i.e., ``data.prediction.models_<cycle>``) can be grouped by functionality
in eight macro-models:

  1. :py:obj:`A/T <at_model>`: gear shifting strategy for automatic transmission,
  2. :py:obj:`electrics <electrics_model>`: vehicle electric components (
     i.e., alternator, service battery, drive battery, and DC/DC converter),
  3. :py:obj:`clutch-torque-converter <clutch_torque_converter_model>`:
     speed model for clutch or torque converter,
  4. :py:obj:`co2_params <co2_params>`: extended willans lines parameters,
  5. :py:obj:`after-treatment <after_treatment_model>`: warm up strategy of after
     treatment,
  6. :py:obj:`engine-coolant-temperature <engine_coolant_temperature_model>`:
     warm up and cooling models of the engine,
  7. :py:obj:`engine-speed <engine_speed_model>`: correlation from velocity to
     engine speed,
  8. :py:obj:`control <control_model>`: start/stop strategy or ECMS.

Model selection
---------------
The default model selection criteria (i.e., when ``enable_selector == False``)
is to use the calibrated models from *WLTP-H* data to predict *WLTP-H* and
*NEDC-H* and from *WLTP_L* data to predict *WLTP-L* and *NEDC-L* (this logic is
applied in type-approval mode).

On the contrary, if the selector is enabled, the function
:py:func:`extract_calibrated_model <co2mpas.core.model.extract_calibrated_model>`
detects/selects the best macro-model for prediction (from *WLTP-H* or *WLTP-L*).
The selection is performed according the model's score, that is the model
capability to reproduce the input data, i.e. a weighted average of all computed
metrics.

In other words, the calibrated models are used to recalculate (**predict**) the
**inputs** of the *WLTP-H* and *WLTP-L* cycles, while the scores derive from
various metrics comparing **inputs** against **predictions**.

.. note::
   A success flag is defined according to upper or lower limits of scores which
   have been defined empirically by the JRC. If a score is outside the model
   fails the calibration and a warning is logged.

.. _substs:

.. |CO2MPAS| replace:: CO\ :sub:`2`\ MPAS
.. |CO2| replace:: CO\ :sub:`2`

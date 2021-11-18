#########
Tutorials
#########
This section explains the functionalities of |co2mpas| GUI through some video
tutorials:

- `Inputs`_

  - `Get input template`_
  - `Synchronize time-series`_
  - `Download demo files`_
- `Run`_

  - `Simulation plan`_
- `Model plot`_

Inputs
======
This section shows the utilities to generate and populate the |co2mpas| input
file.

Get input template
------------------
Check the video to see how to download an empty input excel-file. The generated
file contains the instructions on how to fill the required inputs. For more
information use the command ``co2mpas template -h`` or check the
`link <_build/co2mpas/co2mpas.cli.html#co2mpas-template>`__.

.. raw:: html

    <p>
      <video width="100%" height="%100" controls playsinline preload="metadata">
        <source src="_static/video/download_input_template.mp4" type="video/mp4">
      Your browser does not support the video tag.
      </video>
    </p>

Synchronize time-series
-----------------------
If you have time-series not well synchronized and/or with different sampling
rates, the model might fail. As an aid tool, you may use the ``syncing`` tool to
"synchronize" and "re-sample" your data. To use the tool you should execute the
following steps:

- Generate and download an *empty* input excel-file (see the video).
  For more information use the command ``co2mpas syncing template -h`` or check
  the `link <_build/co2mpas/co2mpas.cli.html#co2mpas-syncing-template>`__.

  .. raw:: html

      <p>
        <video width="100%" height="%100" controls playsinline preload="metadata">
          <source src="_static/video/download_datasync_template.mp4" type="video/mp4">
        Your browser does not support the video tag.
        </video>
      </p>

  .. note::
     All sheets must contain values for columns ``times`` and ``velocities``
     because they are the reference signals used to synchronize the data with
     the theoretical velocity profile.

- Run data synchronization, see the video.
  For more information use the command ``co2mpas syncing sync -h`` or check
  the `link <_build/co2mpas/co2mpas.cli.html#co2mpas-syncing-sync>`__.

  .. raw:: html

      <p>
        <video width="100%" height="%100" controls playsinline preload="metadata">
          <source src="_static/video/run_datasync.mp4" type="video/mp4">
        Your browser does not support the video tag.
        </video>
      </p>

.. note::
   The synchronized signals are saved into the ``synced`` sheet.


Download demo files
-------------------
|co2mpas| contains 4 demo-files that can be used as a starting point to try out:

1. *co2mpas_conventional.xlsx*: conventional vehicle,
2. *co2mpas_simplan.xlsx*: sample simulation plan,
3. *co2mpas_hybrid.xlsx*: hybrid parallel vehicle,
4. *co2mpas_plugin.xlsx*: hybrid plugin vehicle.

Check the video to see how to download them. For more information use the
command ``co2mpas demo -h`` or check the
`link <_build/co2mpas/co2mpas.cli.html#co2mpas-demo>`__.

.. raw:: html

    <p>
      <video width="100%" height="%100" controls playsinline preload="metadata">
        <source src="_static/video/download_demo.mp4" type="video/mp4">
      Your browser does not support the video tag.
      </video>
    </p>

Run
===
This section explains how to run |co2mpas|:

1. Upload excel file/s (see :ref:`previous video <upload_file>`),
2. click run:

.. raw:: html

    <p>
      <video width="100%" height="%100" controls playsinline preload="metadata">
        <source src="_static/video/run_simulation_2.mp4" type="video/mp4">
      Your browser does not support the video tag.
      </video>
    </p>

.. note:: 5 advanced options are available: **use only declaration mode**,
    **hard validation**, **enable selector**, **only summary**, and
    **use custom configuration file**. Flag the box to activate them.

    .. image:: _static/image/advanced_options.png
       :width: 100%
       :alt: |co2mpas| advanced options
       :align: center

3. Get the results  (see the :ref:`previous video <download_results>`).

.. _eng_results:
.. admonition:: Output files.

    - A |co2mpas| output per file, named as ``<timestamp>-<file-name>.xlsx``.
    - A summary file like :ref:`above <ta_results>`.


Physical model configuration file
---------------------------------
The configuration file (.yaml) is used to overwrite the default variables of the
physical model.

Input file:

    - download the conf.yaml template from the GUI as shown in the image below.

    .. image:: _static/image/download_conf.png
       :width: 100%
       :alt: |co2mpas| configuration file
       :align: center

    - modify it according to your parameters.

How to use it:

    - Upload a new configuration file as shown in the picture.

    .. image:: _static/image/conf_1.png
       :width: 100%
       :alt: |co2mpas| configuration file
       :align: center


    - Flag the checkbox *use custom configuration file* under **Advanced options**.

    .. image:: _static/image/conf_2.png
       :width: 100%
       :alt: |co2mpas| configuration file
       :align: center

.. admonition:: conf.yaml

    - the file will be rewritten every time you upload a new one, and it will
      always be named conf.yaml

Simulation plan
---------------
The simulation plan is an input file containing some extra parameters/sheets
with a **scope** ``plan.`` (see :doc:`data naming convention <names>`). It
defines the list of variations (i.e., inputs to be overwritten) that have to be
applied to a base dataset (i.e., a normal input file of |co2mpas|).

The simulation plan can save you time! It is able to calibrate the models
just once and re-use them for other subsequent predictions, where only some
inputs are different (e.g., ``times``, ``velocities``, ``vehicle_mass``, etc.).

To run the simulation plan upload it as an input file, and run it as described in
the previous section.

.. admonition:: Output files

    - A |co2mpas| output per **file** like :ref:`above <eng_results>`.
    - A |co2mpas| output per **variation**, named as
      ``<timestamp>-<variation-id>-<file-name>.xlsx``.
    - A summary file like :ref:`above <ta_results>`.

Model plot
==========
This section shows the utility to investigate the |co2mpas| model. For more
information check :doc:`model` and :doc:`api`.

.. raw:: html

    <p>
      <video width="100%" height="%100" controls playsinline preload="metadata">
        <source src="_static/video/model_plot.mp4" type="video/mp4">
      Your browser does not support the video tag.
      </video>
    </p>

.. |co2mpas| replace:: CO\ :sub:`2`\ MPAS

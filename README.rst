
.. image:: https://rbturnbull.github.io/WiDiT/_images/WiDiT-Banner.png
   :alt: WiDiT banner
   :align: center

.. start-badges

|pypi badge| |testing badge| |coverage badge| |docs badge| |black badge|

.. |pypi badge| image:: https://img.shields.io/pypi/v/widit.svg?color=blue
    :target: https://pypi.org/project/widit/

.. |testing badge| image:: https://github.com/rbturnbull/widit/actions/workflows/testing.yml/badge.svg
    :target: https://github.com/rbturnbull/widit/actions

.. |docs badge| image:: https://github.com/rbturnbull/widit/actions/workflows/docs.yml/badge.svg
    :target: https://rbturnbull.github.io/widit
    
.. |black badge| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    
.. |coverage badge| image:: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/rbturnbull/f68582048631310754cc9719e4fc7cf9/raw/coverage-badge.json
    :target: https://rbturnbull.github.io/widit/coverage/

    
.. end-badges

.. start-quickstart


WiDiT is a SwinIR-style DiT backbone that unifies **2D images** and **3D volumes**
with N-D windowed attention, optional Swin shifts, and AdaLN-Zero conditioning.

- Single model class: ``widit.models.WiDiT``
- Also includes a configurable ``widit.models.Unet`` for 2D/3D U-Net baselines
- Optional timestep conditioning (pass ``timestep=None`` if unused)
- Shared blocks for 2D/3D via N-D window partitioning
- Presets for quick experiments in both 2D and 3D

Installation
----------------

Install using pip:

.. code-block:: bash

    pip install widit

Or

.. code-block:: bash

    pip install git+https://github.com/rbturnbull/widit.git

WiDiT depends on ``torch``.

.. warning::

    WiDiT is currently in alpha testing phase. More updates are coming soon.


Quick Start (2D)
----------------

.. code-block:: python

   import torch
   from widit.models import WiDiT

   # Example: 2D RGB input & conditioning (e.g., low-res guidance)
   N, C, H, W = 2, 3, 128, 96
   x      = torch.randn(N, C, H, W)
   cond   = torch.randn_like(x)
   t      = torch.randint(0, 1000, (N,), dtype=torch.long)  # optional

   model = WiDiT(
       spatial_dim=2,
       patch_size=2,             # must divide H and W
       in_channels=C,
       hidden_size=256,          # must be divisible by num_heads and even
       depth=6,
       num_heads=8,
       window_size=8,            # can be int or (wh, ww)
       mlp_ratio=4.0,
       learn_sigma=True,         # output channels = 2*C if True
       use_conditioning=True,    # expect a conditioning image
   )

   # NEW CALL SIGNATURE:
   # forward(input, timestep=None, *, conditioned=None)
   y = model(x, t, conditioned=cond)     # (N, 2*C, H, W) if learn_sigma=True


Quick Start (3D)
----------------

.. code-block:: python

   import torch
   from widit.models import WiDiT

   # Example: 3D single-channel volumes
   N, C, D, H, W = 1, 1, 64, 64, 48
   x    = torch.randn(N, C, D, H, W)
   cond = torch.randn_like(x)

   model = WiDiT(
       spatial_dim=3,
       patch_size=2,             # must divide D/H/W
       in_channels=C,
       hidden_size=256,
       depth=4,
       num_heads=8,
       window_size=(4, 4, 4),    # can be int or (wd, wh, ww)
       mlp_ratio=4.0,
       learn_sigma=False,        # output channels = C if False
       use_conditioning=True,
   )

   y = model(x, timestep=None, conditioned=cond)  # (N, C, D, H, W)


Unconditioned Image Path (no second image)
------------------------------------------

.. code-block:: python

   import torch
   from widit.models import WiDiT

   N, C, H, W = 2, 3, 128, 96
   x = torch.randn(N, C, H, W)
   t = torch.randint(0, 1000, (N,))

   model = WiDiT(
       spatial_dim=2,
       in_channels=C,
       hidden_size=256,
       depth=4,
       num_heads=8,
       patch_size=2,
       window_size=8,
       learn_sigma=True,
       use_conditioning=False,       # <-- no conditioning image expected
   )

   # Do NOT pass `conditioned` when use_conditioning=False
   y = model(x, t)  # (N, 2*C, H, W)


Presets
-------

Presets provide ready-made configurations for common model sizes (2D & 3D), all
using ``patch_size=2`` and Swin-style window attention:

.. code-block:: python

   from widit.models import PRESETS
   import torch

   # 2D: B, M, L, XL
   model_2d = PRESETS["WiDiT2D-L"](in_channels=3, learn_sigma=True)

   # 3D: B, M, L, XL
   model_3d = PRESETS["WiDiT3D-M"](in_channels=1, learn_sigma=False)

   # Example inputs
   x2d = torch.randn(1, 3, 64, 48)
   c2d = torch.randn_like(x2d)
   t2d = torch.randint(0, 1000, (1,))

   x3d = torch.randn(1, 1, 32, 32, 24)
   c3d = torch.randn_like(x3d)
   t3d = torch.randint(0, 1000, (1,))

   # Run
   y2d = model_2d(x2d, t2d, conditioned=c2d)
   y3d = model_3d(x3d, timestep=None, conditioned=c3d)


Loading Models
--------------

Use ``load_model`` to load a saved WiDiT or Unet checkpoint. It infers the
correct class from the stored config:

.. code-block:: python

   from widit import load_model

   model = load_model("path/to/checkpoint.pt")




.. end-quickstart


Credits
==================================

.. start-credits

`Robert Turnbull <https://robturnbull.com>`_ - Melbourne Data Analytics Platform (MDAP), The University of Melbourne

.. end-credits

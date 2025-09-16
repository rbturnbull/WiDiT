
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
- Optional timestep conditioning (pass ``timestep=None`` if unused)
- Shared blocks for 2D/3D via N-D window partitioning
- Presets for quick experiments in both 2D and 3D

Installation
==================================

Install using pip:

.. code-block:: bash

    pip install widit

Or

.. code-block:: bash

    pip install git+https://github.com/rbturnbull/widit.git

WiDiT depends on ``torch`` and ``timm`` (for the 2D patch embedding path).

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
       input_size=(H, W),        # kept for API parity; not required at forward
       patch_size=2,             # must divide H and W
       in_channels=C,
       hidden_size=256,          # must be divisible by num_heads and even
       depth=6,
       num_heads=8,
       window_size=8,            # can be int or (wh, ww)
       mlp_ratio=4.0,
       learn_sigma=True,         # output channels = 2*C if True
   )

   # Timestep is optional; pass None to disable conditioning
   y = model(x, cond, t)         # (N, 2*C, H, W) if learn_sigma=True


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
       input_size=(D, H, W),
       patch_size=2,             # must divide D/H/W
       in_channels=C,
       hidden_size=256,
       depth=4,
       num_heads=8,
       window_size=(4, 4, 4),    # can be int or (wd, wh, ww)
       mlp_ratio=4.0,
       learn_sigma=False,        # output channels = C if False
   )

   y = model(x, cond, timestep=None)  # (N, C, D, H, W)


Presets
-------

Presets provide ready-made configurations for common model sizes (2D & 3D), all
using ``patch_size=2`` and Swin-style window attention:

.. code-block:: python

   from widit.models import PRESETS

   # 2D: B, M, L, XL
   model_2d = PRESETS["WiDiT-L/2"](in_channels=3, learn_sigma=True)

   # 3D: B, M, L, XL
   model_3d = PRESETS["WiDiT3D-M/2"](in_channels=1, learn_sigma=False)

   # Run
   y2d = model_2d(x2d, cond2d, timestep=None)
   y3d = model_3d(x3d, cond3d, timestep=torch.randint(0, 1000, (x3d.shape[0],)))


API Overview
------------

.. code-block:: python

   WiDiT(
       *,
       spatial_dim: int,                          # 2 (images) or 3 (volumes)
       input_size: int | Sequence[int] | None = None,
       patch_size: int | Sequence[int] = 2,       # per-axis tuple allowed
       in_channels: int = 1,
       hidden_size: int = 768,                    # even; divisible by num_heads
       depth: int = 12,
       num_heads: int = 12,
       window_size: int | Sequence[int] = 8,      # per-axis tuple allowed
       mlp_ratio: float = 4.0,
       learn_sigma: bool = True,
   )

   forward(
       input_tensor:       torch.Tensor,          # (N, C, *spatial)
       conditioned_tensor: torch.Tensor,          # (N, C, *spatial), same shape as input_tensor
       timestep:           torch.Tensor | None = None,  # (N,) or None
   ) -> torch.Tensor                              # (N, out_channels, *spatial)

**Shapes & contracts**

- ``*spatial`` is ``(H, W)`` for 2D and ``(D, H, W)`` for 3D.
- ``patch_size`` must evenly divide each spatial dimension.
- ``window_size`` can be an int or a per-axis tuple; internal padding ensures
  full windows (removed before returning).
- ``hidden_size`` must be **even** (split across the two patch embedders) and divisible by ``num_heads``.
- If ``learn_sigma=True``, output channels = ``2 * in_channels`` (mean + sigma style).

**Conditioning**

- ``timestep`` is **optional**. Pass ``None`` to disable AdaLN conditioning (the
  block falls back to standard LN + residual).
- If provided, the model uses ``widit.timesteps.TimestepEmbedder`` to produce
  a per-sample vector projected to the token dimension.


Building Blocks
~~~~~~~~~~~~~~~

These are used internally, but you can also import them for custom stacks.

- ``widit.blocks.WiDiTBlock`` – N-D windowed MSA + MLP with AdaLN-Zero
- ``widit.blocks.WiDiTFinalLayer`` – final projection head with AdaLN-Zero
- ``widit.patch.PatchEmbed`` – unified 2D/3D patch embedding
- ``widit.timesteps.TimestepEmbedder`` – sinusoidal → MLP conditioning

All of the above expose ``init_weights()`` so the model can initialize components
cleanly (adaLN-Zero policy for blocks & head; Xavier for projections; Normal for
timestep MLP weights).


Training Snippet
----------------

.. code-block:: python

   import torch
   from torch.optim import AdamW
   from widit.models import WiDiT

   device = "cuda" if torch.cuda.is_available() else "cpu"

   model = WiDiT(
       spatial_dim=2,
       in_channels=3,
       hidden_size=256,
       depth=6,
       num_heads=8,
       patch_size=2,
       window_size=8,
       learn_sigma=True,
   ).to(device)

   opt = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

   for step in range(100):
       x    = torch.randn(8, 3, 128, 96, device=device)
       cond = torch.randn_like(x)
       t    = torch.randint(0, 1000, (x.shape[0],), device=device)

       y = model(x, cond, t)                      # (N, 6, H, W) here (mean+sigma for C=3)
       target = torch.randn_like(y)

       loss = torch.nn.functional.mse_loss(y, target)
       opt.zero_grad(set_to_none=True)
       loss.backward()
       opt.step()


Tips & Gotchas
--------------

- **Patch size equality in unpatchify**: currently the unpatchify path enforces
  equal patch size along all axes (e.g., ``patch_size=2`` or ``(2,2,2)``). Mixed
  per-axis patch sizes for output reconstruction are not supported yet.
- **Token grid divisibility**: ensure every spatial dimension is divisible by
  ``patch_size``. Window attention will pad internally to complete windows and
  crop back, but patch embedding is stride-based.
- **Timestep optional**: pass ``timestep=None`` to run the model without diffusion
  conditioning (AdaLN defaults reduce to a vanilla transformer residual path).
- **Mixed precision**: standard AMP (``torch.cuda.amp``) works out-of-the-box.


Reference Shapes
----------------

**2D**

- Input:  ``(N, C, H, W)``
- Output: ``(N, 2*C, H, W)`` if ``learn_sigma=True``, else ``(N, C, H, W)``

**3D**

- Input:  ``(N, C, D, H, W)``
- Output: ``(N, 2*C, D, H, W)`` if ``learn_sigma=True``, else ``(N, C, D, H, W)``


.. end-quickstart


Credits
==================================

.. start-credits

`Robert Turnbull <https://robturnbull.com>`_ - Melbourne Data Analytics Platform (MDAP), The University of Melbourne

.. end-credits


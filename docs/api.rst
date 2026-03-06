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
       use_conditioning: bool = True,             # expect a second image unless set False
   )

   forward(
       input_tensor: torch.Tensor,                # (N, C, *spatial)
       timestep: torch.Tensor | None = None,      # (N,) or None
       *,                                          # keyword-only from here
       conditioned: torch.Tensor | None = None,   # (N, C, *spatial) if use_conditioning=True
   ) -> torch.Tensor                              # (N, out_channels, *spatial)

**Shapes & contracts**

- ``*spatial`` is ``(H, W)`` for 2D and ``(D, H, W)`` for 3D.
- ``patch_size`` must evenly divide each spatial dimension.
- ``window_size`` can be an int or a per-axis tuple; internal padding ensures
  full windows (removed before returning).
- ``hidden_size`` must be **even** (split across the two patch embedders when
  ``use_conditioning=True``) and divisible by ``num_heads``.
- If ``learn_sigma=True``, output channels = ``2 * in_channels`` (mean + sigma style).
- If ``use_conditioning=True``, you **must** pass ``conditioned=...`` to ``forward``.
  If ``use_conditioning=False``, passing ``conditioned`` will raise an assertion.

**Conditioning**

- ``timestep`` is **optional**. Pass ``None`` to disable AdaLN conditioning (the
  blocks reduce to standard LN + residual).
- If provided, the model uses ``widit.timesteps.TimestepEmbedder`` to produce
  a per-sample vector projected to the token dimension.


Building Blocks
~~~~~~~~~~~~~~~

These are used internally, but you can also import them for custom stacks.

- ``widit.blocks.WiDiTBlock`` – N-D windowed MSA + MLP with AdaLN-Zero
- ``widit.blocks.WiDiTFinalLayer`` – final projection head with AdaLN-Zero
- ``widit.patch.PatchEmbed`` – unified 2D/3D patch embedding (with ``init_weights()``)
- ``widit.timesteps.TimestepEmbedder`` – sinusoidal → MLP conditioning (with ``init_weights()``)

All of the above expose ``init_weights()`` so the model can initialize components
cleanly (adaLN-Zero policy for blocks & head; Xavier for projections; Normal for
timestep MLP weights).

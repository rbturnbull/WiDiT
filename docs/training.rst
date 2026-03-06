Training
==============

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
       use_conditioning=True,
   ).to(device)

   opt = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

   for step in range(100):
       x    = torch.randn(8, 3, 128, 96, device=device)
       cond = torch.randn_like(x)
       t    = torch.randint(0, 1000, (x.shape[0],), device=device)

       y = model(x, t, conditioned=cond)          # (N, 6, H, W) here (mean+sigma for C=3)
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
- **Conditioning toggle**: if you don’t have a conditioning image, set
  ``use_conditioning=False`` and call ``model(x, timestep)``






Reference Shapes
----------------

**2D**

- Input:  ``(N, C, H, W)``
- Output: ``(N, 2*C, H, W)`` if ``learn_sigma=True``, else ``(N, C, H, W)``

**3D**

- Input:  ``(N, C, D, H, W)``
- Output: ``(N, 2*C, D, H, W)`` if ``learn_sigma=True``, else ``(N, C, D, H, W)``



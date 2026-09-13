"""Small signed/diluted XY example; no generated files."""

import numpy as np
from peapods import XY

occupation = np.random.default_rng(12).random((8, 8)) > 0.2
model = XY(
    (8, 8),
    couplings="bimodal",
    temperatures=[0.8, 1.2, 2.0],
    n_disorder=4,
    occupation=occupation,
    seed=42,
)
result = model.sample(
    2048,
    collect_blocks=True,
    displacements=[[1, 0], [2, 0]],
    vortices=True,
    pt_interval=1,
    pt_schedule="full_ladder",
)
print("Physical energy/site:", result["energies"])
print("Signed correlations:", result["correlations"])
print("Intact plaquette fraction:", result["intact_plaquette_fraction"])

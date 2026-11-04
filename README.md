# ExciView

**BSE/Excited State Analysis Tools for FHI-aims**

**Implemented by:** Ruiyi Zhou (ETH Zurich) ruiyi.zhou@phys.chem.ethz.ch

ExciView is a Python-based toolkit designed to analyze excited state wavefunctions (excitons) calculated using the Bethe-Salpeter Equation (BSE) formalism in **FHI-aims**. It bridges the gap between reciprocal space (k-space) eigenvector weights, real-space atomic contributions, and complex 3D volumetric wavefunctions.

## 🚀 Key Features

*   **Efficient Binary I/O:** Reads specific eigenvectors directly from large ELSI binary files without loading the full matrix into memory.
*   **Reciprocal Space Analysis:** Analyzes momentum distribution ($k$-weights) and identifies dominant orbital transitions (e.g., HOMO/LUMO character) across the Brillouin zone.
*   **Real-Space Projection (Mulliken):** Performs decomposition analysis for particle (electron) and hole densities in systems with Periodic Boundary Conditions, including per-atom orbital angular momentum ($s, p, d, f$) breakdown.
*   **Average Volumetric Density:** Generates `.cube` files for the average hole and electron densities by performing weighted incoherent sums of band densities.
*   **Conditional Wavefunction Analysis:** Calculates the correlated electron density distribution for a specific *fixed* hole position ($\rho_e(\mathbf{r} | \mathbf{r}_h)$) by performing coherent wavefunction summation.
*   **Smart Thresholding:** Optimizes computational cost by identifying dominant $k$-points/bands and generating optimized input snippets for FHI-aims.

## 📋 Workflow

The tool operates via an interactive menu with four main analysis pipelines:

### 1. Atomic Population Pipeline (Mulliken)
*   **Phase 1 (Screening):** Scans the binary BSE file to identify dominant $k$-points ($W_k > \epsilon$) and generates a `mulliken_snippet.in` for FHI-aims.
*   **Phase 2 (Analysis):** Maps the resulting FHI-aims `band_mulliken` outputs back to the BSE weights to compute the final spatial distribution and orbital character.

### 2. Average Density Pipeline (Volumetric)
*   **Phase 3 (Cube Generation):** Identifies specific (k-point, band) pairs that contribute significantly to the hole/electron and generates a `cube_snippet.in` requesting `eigenstate_density`.
*   **Phase 4 (Summation):** Reads the requested `.cube` files (validated against `control.in`) and sums them according to their BSE weights to produce `avg_hole.cube` and `avg_elec.cube`.

### 3. Brillouin Zone & Band Analysis
*   **Phase 5 (Statistics):** Performs a pure statistical analysis of the BSE eigenvector to report the dominant $k$-points and the dominant band transitions (e.g., HOMO-1 $\to$ LUMO) responsible for the excitation.

### 4. Conditional Density Pipeline
*   **Phase 6 (Input Gen):** Generates input requests for raw `eigenstate` wavefunctions (preserving phase information) for relevant bands.
*   **Phase 7 (Coherent Sum):** Asks for a fixed hole coordinate $\mathbf{r}_h$, extracts hole amplitudes, computes mixing coefficients, and constructs the conditional electron density $\rho_{cond}(\mathbf{r})$.

## 🛠 Roadmap / To-Do

- [x] **Volumetric Plotting:** Generate `.cube` files for average hole/electron density visualization.
- [x] **Conditional Density:** Coherent complex-wavefunction summation with a fixed hole position (requires a compatible orbital cube producer).
- [ ] **Molecular Analysis:** Support for finite systems (Molecules) covering LR-TDDFT, BSE, and full-frequency GW analysis.

## 📦 Requirements
*   Python 3.x
*   NumPy
*   SciPy
*   **ASE (Atomic Simulation Environment)** (Required for Cube file operations)

## ✉️ Contact
For questions or issues, please contact **Ruiyi Zhou** (ruiyi.zhou@phys.chem.ethz.ch) at ETH Zurich.

## LibBSE integration

`libbse_to_exciview.py` imports LibBSE **TDA, regular SCF-grid** calculations.
It joins the contiguous electron–hole pair intervals in numeric MPI rank order,
checks rank sizes and state norms, and stores `(state, k, v, c)` complex amplitudes,
Hartree energies, actual k coordinates, band IDs, gauge, and source hashes in NPZ.
It supports all TDA spin channels; full-BSE X/Y and fine-grid imports are not
implemented. The existing ELSI input remains supported.

```bash
python -m pip install -r requirements.txt
python tools/libbse_to_exciview.py /path/to/bse_run --v-start 11 --c-start 15 -o excitons.npz
python BSE_tools.py
```

Enter `excitons.npz` at the filename prompt. **State Index is zero based** (0 is
the lowest state); absolute band IDs and cube k-point IDs are one based.
For Si 3×3×3 with bands 11–14 / 15–17, use Nv=4, Nc=3, Nk=27 and Kx=Ky=Kz=3.
The archive's dimensions, band IDs and k coordinates are validated rather than
inferred from filename order. Thresholds are percentages of BSE weight, not a
requested cumulative accuracy. Threshold 0 requests the full band window.

The LibBSE importer also reads original `bandmlk*.out` files containing multiple
k points per band segment (for example, nine three-point segments for a 3×3×3
mesh). No splitting or intermediate `mulliken_snippet.in` is needed. It searches,
in order, the BSE run directory, `input_dir`, and sibling directories named
`DFT_muliken`, `DFT_mulliken`, or `aims_mulliken_333`. For another location, pass
`--mulliken-dir /path/to/mulliken_run` (relative paths are relative to the shell's
working directory). With no detected directory the import remains amplitude-only.

Each block is matched by its periodic k coordinates to the BSE grid, independently
of path labels and file order. The full BSE k grid and requested band/atom coverage
are required; repeated endpoints are accepted only if their populations agree.
Missing points, nonfinite populations and conflicting duplicates fail the import.
Total and angular-momentum populations and source hashes are embedded in NPZ.
Menu 2 then only asks for the archive, state index, Nv, Nc and Nk; it uses the
stored band IDs and populations automatically. Existing archives without Mulliken
data and legacy ELSI files retain the snippet/file-pattern workflow. Re-import an
existing BSE run to add Mulliken data to its archive. Matching coordinates does
not align independently diagonalized degenerate orbitals; the physical band
gauge consistency requirement still applies.

All seven menu entries are enabled. Conditional input generation requests **both**
`eigenstate` and `eigenstate_imag`; analysis forms
`abs(sum_kvc A[k,v,c] * conj(psi[v,k](r_h)) * psi[c,k](r))**2`.
Missing/ambiguous cubes, incompatible grids, origins or geometries cause an error.
The output retains the original cube origin. The hole coordinate is in Angstrom
and must lie inside the supplied cube: wrapping a Bloch orbital without its phase
is not valid. The containing voxel is sampled (no interpolation). Conditional
output is raw squared amplitude, not a unit-normalized conditional probability.
Average density and Mulliken outputs are normalized over retained contributions;
the captured weights/populations are printed in the logs.

Complex orbital cubes must come from the **same native band gauge** as the BSE
calculation. Densities alone cannot reconstruct these phases, and a separate
SCF/band diagonalization can rotate degenerate orbitals. Some FHI-aims versions
explicitly disable off-Gamma orbital cubes and have an imaginary-buffer bug;
Phase 6 writes requests but cannot make an unsupported producer execute them.
The local Si integration fixture includes a separately built scalar-relativistic
producer repair and Bloch-translation tests. It does not enable general SOC cubes.

Regression tests (all seven menus, independent analytic sums, gauge covariance,
ELSI real/complex sparse I/O, malformed inputs and unequal MPI partitions):

```bash
python -m unittest discover -s tests -v
```

Local complete Si fixture: `/work/users/l/s/lsr/LibBSE/1_LibBSE_test/17_post`.
See its `README.md`, `verification.json`, `analysis/phase*.log`, and recorded
producer patches for measured results and reproduction commands.

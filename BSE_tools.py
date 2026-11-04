import struct
import numpy as np
import scipy.sparse as sp
import sys
import os
import re
import glob

# Try to import ASE (required for Phases 4 and 7)
try:
    from ase.io.cube import read_cube, write_cube
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False

# =============================================================================
#                                CORE FUNCTIONS
# =============================================================================

def read_elsi_state(filename, state_index):
    """Read a zero-based state from ELSI or the LibBSE import NPZ archive."""
    if str(filename).endswith('.npz'):
        with np.load(filename, allow_pickle=False) as archive:
            coeffs = archive['coefficients']
            if coeffs.ndim != 4 or not 0 <= state_index < len(coeffs):
                raise ValueError('Invalid coefficient shape or zero-based state index')
            vector = coeffs[state_index].reshape(-1)
            if not np.isfinite(vector).all():
                raise ValueError('Non-finite coefficients')
            return sp.csc_matrix(vector[:, None])
    fmt_i8 = "q"; fmt_i4 = "i"; fmt_d  = "d"
    size_i8 = 8; size_i4 = 4; size_d  = 8
    
    with open(filename, "rb") as f:
        header_data = f.read(128)
        if len(header_data) != 128:
            raise ValueError('Truncated ELSI header')
        header = struct.unpack('<' + fmt_i8 * 16, header_data)
        is_complex = (header[2] != 0); n_basis = header[3]; nnz_total = header[5]
        if header[0] != 170915 or header[2] not in (0, 1) or n_basis < 1 or nnz_total < 0:
            raise ValueError('Invalid ELSI header')
        if os.fstat(f.fileno()).st_size != 128 + n_basis*8 + nnz_total*(4 + (16 if is_complex else 8)):
            raise ValueError('ELSI file size does not match header')
        
        if state_index < 0 or state_index >= n_basis:
            raise ValueError(f"State index {state_index} is out of bounds (Max: {n_basis-1})")

        offset_col_ptr = 128
        offset_row_idx = offset_col_ptr + (n_basis * size_i8)
        offset_values  = offset_row_idx + (nnz_total * size_i4)
        
        f.seek(offset_col_ptr + (state_index * size_i8))
        ptr_start = struct.unpack('<q', f.read(size_i8))[0]
        ptr_end = nnz_total + 1 if state_index == n_basis - 1 else struct.unpack('<q', f.read(size_i8))[0]
        if not 1 <= ptr_start <= ptr_end <= nnz_total+1:
            raise ValueError('Invalid ELSI column pointers')
        n_elements = ptr_end - ptr_start
        
        if n_elements == 0:
            return sp.csc_matrix((n_basis, 1), dtype=complex if is_complex else float)

        data_offset = ptr_start - 1
        f.seek(offset_row_idx + (data_offset * size_i4))
        row_indices = np.array(struct.unpack('<' + fmt_i4 * n_elements, f.read(n_elements * size_i4))) - 1
        if np.any(row_indices < 0) or np.any(row_indices >= n_basis):
            raise ValueError('Invalid ELSI row indices')
        
        offset_v = offset_values + (data_offset * (16 if is_complex else 8))
        f.seek(offset_v)
        if not is_complex:
            values = np.array(struct.unpack('<' + fmt_d * n_elements, f.read(n_elements * 8)))
        else:
            raw = struct.unpack('<' + fmt_d * (n_elements * 2), f.read(n_elements * 16))
            values = np.array(raw[0::2]) + 1j * np.array(raw[1::2])
        if not np.isfinite(values).all():
            raise ValueError('Non-finite ELSI values')

    col_ptr = np.array([0, n_elements])
    return sp.csc_matrix((values, row_indices, col_ptr), shape=(n_basis, 1))

def parse_aims_mulliken_file(filename):
    """Parses a single FHI-aims band_mulliken output file."""
    mulliken_data = {} 
    current_band = None
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.split()
            if not parts: continue
            if parts[0] == "State" and len(parts) == 2 and parts[1].isdigit():
                current_band = int(parts[1])
                mulliken_data[current_band] = {}
                continue
            if current_band is not None and parts[0].isdigit() and int(parts[0]) == current_band:
                try:
                    atom_id = int(parts[3])
                    total_val = float(parts[4])
                    # Handle variable orbital lengths (s, p, d...)
                    orb_vals = np.array([float(x) for x in parts[5:]])
                    mulliken_data[current_band][atom_id] = {'total': total_val, 'orbitals': orb_vals}
                except (ValueError, IndexError): continue
    except FileNotFoundError: return None
    return mulliken_data

def get_bse_weights(filename, state_idx, nk, nv, nc):
    """Returns probability weights tensor (Nk, Nv, Nc)."""
    if min(nk, nv, nc) <= 0:
        raise ValueError('Nk, Nv and Nc must be positive')
    if str(filename).endswith('.npz'):
        with np.load(filename, allow_pickle=False) as archive:
            if archive['coefficients'].shape[1:] != (nk, nv, nc):
                raise ValueError('Nk/Nv/Nc do not match the LibBSE archive')
    sparse_vec = read_elsi_state(filename, state_idx)
    dense_vec = sparse_vec.toarray().flatten()
    expected_size = nk * nv * nc
    if dense_vec.shape[0] != expected_size:
        raise ValueError(f"Dimension mismatch! File: {dense_vec.shape[0]}, Expected: {expected_size}")
    coeffs = dense_vec.reshape((nk, nv, nc))
    return np.abs(coeffs)**2

def parse_snippet_for_mapping(snippet_filename):
    """Reads mapping from input snippet."""
    k_indices = []
    try:
        with open(snippet_filename, 'r') as f:
            for line in f:
                if line.strip().startswith("output band_mulliken"):
                    match = re.search(r"K_(\d+)", line)
                    if match:
                        k_indices.append(int(match.group(1)))
    except FileNotFoundError:
        print(f"[Error] Mapping file not found: {snippet_filename}")
        return None
    return k_indices

def parse_control_for_cubes(control_file="cube_snippet.in"):
    """Parses control.in to find which cubes were actually requested."""
    valid_cubes = set()
    current_band = None
    try:
        with open(control_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            parts = line.split()
            if not parts: continue
            if parts[0] == "output" and "cube" in parts and ("eigenstate_density" in parts or "eigenstate" in parts):
                try: current_band = int(parts[-1])
                except ValueError: current_band = None
            elif "cube" in parts and "kpoint" in parts:
                try:
                    k_idx = int(parts[-1])
                    if current_band is not None: valid_cubes.add((k_idx, current_band))
                except ValueError: pass
    except FileNotFoundError:
        print(f"[Error] Could not read {control_file}")
        return None
    return valid_cubes

def get_k_coords(k_index, nkx, nky, nkz, filename=None):
    if filename is not None and str(filename).endswith('.npz'):
        with np.load(filename, allow_pickle=False) as archive:
            if not np.array_equal(archive['kmesh'], [nkx,nky,nkz]):
                raise ValueError('Kx/Ky/Kz do not match the LibBSE archive')
            return tuple(archive['kpoints'][k_index])
    if nkx * nky * nkz == 0: return (0.0, 0.0, 0.0)
    iz = k_index % nkz; iy = (k_index // nkz) % nky; ix = k_index // (nkz * nky)
    return (ix/nkx if nkx>1 else 0, iy/nky if nky>1 else 0, iz/nkz if nkz>1 else 0)

def get_grid_val_at_coord(data, atoms, coord, origin=None):
    """Sample the containing voxel; Cartesian coordinate and origin are in Angstrom.

    Do not wrap complex Bloch wavefunctions: crossing a unit cell changes phase.
    """
    cell = atoms.get_cell()
    frac = np.dot(np.asarray(coord) - (np.zeros(3) if origin is None else origin), np.linalg.inv(cell))
    if np.any(frac < -1e-8) or np.any(frac >= 1.0):
        raise ValueError('Fixed hole lies outside the cube; periodic Bloch wrapping requires its phase')
    shape = np.array(data.shape)
    indices = np.floor(np.maximum(frac, 0) * shape + 1e-10).astype(int)
    return data[indices[0], indices[1], indices[2]]


def load_cube_target(pattern, band, kpoint, reference=None):
    files = sorted(glob.glob(pattern.format(band, kpoint)))
    if len(files) != 1:
        raise ValueError(f'Expected exactly one cube for band {band}, k {kpoint}; found {len(files)}')
    with open(files[0]) as stream:
        content = read_cube(stream, read_data=True)
    if not np.isfinite(content['data']).all():
        raise ValueError(f'Non-finite cube: {files[0]}')
    if reference is not None:
        if (content['data'].shape != reference['data'].shape
            or not np.allclose(content['origin'], reference['origin'], atol=1e-7, rtol=0)
            or not np.allclose(content['atoms'].cell, reference['atoms'].cell, atol=1e-7, rtol=0)
            or not np.array_equal(content['atoms'].numbers, reference['atoms'].numbers)
            or not np.allclose(content['atoms'].positions, reference['atoms'].positions, atol=1e-7, rtol=0)):
            raise ValueError(f'Incompatible cube grid/geometry: {files[0]}')
    return content


def check_band_window(filename, v_start, c_start):
    if str(filename).endswith('.npz'):
        with np.load(filename, allow_pickle=False) as archive:
            if (v_start, c_start) != (int(archive['v_start']), int(archive['c_start'])):
                raise ValueError('Absolute band IDs do not match the LibBSE archive')

# =============================================================================
#                                WORKFLOWS
# =============================================================================

def workflow_generate_inputs_mulliken():
    print("\n--- PHASE 1: Generate Mulliken Inputs ---")
    fname = input("Eigenvectors filename (ELSI / LibBSE NPZ): ").strip()
    if not os.path.exists(fname): raise FileNotFoundError(fname)
    
    state_idx = int(input("State Index: "))
    nv = int(input("Nv: ")); nc = int(input("Nc: ")); nk = int(input("Nk: "))
    nkx = int(input("Kx: ")); nky = int(input("Ky: ")); nkz = int(input("Kz: "))
    
    weights = get_bse_weights(fname, state_idx, nk, nv, nc)

    k_weights = np.sum(weights, axis=(1, 2))
    norm = np.sum(k_weights)
    if norm > 0: k_weights /= norm

    thresh = float(input("Threshold %: ")) / 100.0
    out_file = "mulliken_snippet.in"
    count = 0
    with open(out_file, "w") as f:
        f.write(f"# Mulliken Snippet | State {state_idx} | Thresh {thresh*100}%\n")
        for i, w in enumerate(k_weights):
            if w >= thresh:
                count += 1
                kx, ky, kz = get_k_coords(i, nkx, nky, nkz, fname)
                label = f"K_{i}"
                line = (f"output band_mulliken {kx:.12f} {ky:.12f} {kz:.12f} "
                        f"{kx:.12f} {ky:.12f} {kz:.12f} 2 {label} {label}")
                f.write(line + "\n")
    print(f"Success! Wrote {count} lines to {out_file}")

def workflow_analyze_spatial():
    print("\n--- PHASE 2: Spatial Analysis (Mulliken) ---")
    fname = input("Eigenvectors filename (ELSI / LibBSE NPZ): ").strip()
    if not os.path.exists(fname): raise FileNotFoundError(fname)
    state_idx = int(input("State Index: "))
    nv = int(input("Nv: ")); nc = int(input("Nc: ")); nk = int(input("Nk: "))
    
    weights = get_bse_weights(fname, state_idx, nk, nv, nc)
    embedded = None
    if str(fname).endswith('.npz'):
        with np.load(fname, allow_pickle=False) as archive:
            if 'mulliken_populations' in archive:
                embedded = archive['mulliken_populations']
                band_ids = archive['mulliken_band_ids']
                stored_atoms = archive['mulliken_atom_ids']
                v_start, c_start = int(archive['v_start']), int(archive['c_start'])
        if embedded is not None:
            check_band_window(fname, v_start, c_start)
            if (embedded.ndim != 4 or embedded.shape[:3] != (nk, nv+nc, len(stored_atoms))
                    or embedded.shape[3] < 2 or not np.isfinite(embedded).all()
                    or not np.array_equal(band_ids, np.arange(v_start, c_start+nc))
                    or not np.array_equal(stored_atoms, np.unique(stored_atoms))
                    or np.any(stored_atoms < 1) or np.any(stored_atoms != stored_atoms.astype(int))):
                raise ValueError('Invalid embedded Mulliken dimensions/bands/atoms')
            k_indices = range(nk)
            print(f'Using embedded Mulliken data: {nk} k points, bands {v_start}–{c_start+nc-1}')
    if embedded is None:
        snippet_file = input("Snippet file (default: mulliken_snippet.in): ").strip()
        if not snippet_file: snippet_file = "mulliken_snippet.in"
        k_indices = parse_snippet_for_mapping(snippet_file)
        if not k_indices: raise ValueError("No k points in Mulliken mapping")
        v_start = int(input("First Valence Band ID: "))
        c_start = int(input("First Conduction Band ID: "))
        check_band_window(fname, v_start, c_start)
        pat = input("File Pattern (e.g. bandmlk{}.out): ")
        offset = int(input("Start Offset (e.g. 1001): "))

    hole_pop = None; elec_pop = None; atom_ids = []
    hole_atom_orbs = {}; elec_atom_orbs = {}
    MAX_ORBS = max(6, embedded.shape[3]-1) if embedded is not None else 6
    max_orb = 0; files_proc = 0

    print("Processing...")
    for i, k_idx in enumerate(k_indices):
        if embedded is None:
            m_file = pat.format(offset + i)
            m_data = parse_aims_mulliken_file(m_file)
        else:
            m_file = fname
            m_data = {int(band): {int(atom): {
                'total': embedded[k_idx, ib, ia, 0],
                'orbitals': embedded[k_idx, ib, ia, 1:],
            } for ia, atom in enumerate(stored_atoms)} for ib, band in enumerate(band_ids)}
        if not m_data: raise ValueError(f'Missing or empty Mulliken file: {m_file}')
        if not 0 <= k_idx < nk: raise ValueError('Mulliken k index out of range')
        missing = set(range(v_start,v_start+nv)) | set(range(c_start,c_start+nc))
        if not missing.issubset(m_data): raise ValueError(f'Missing bands in {m_file}')
        files_proc += 1
        
        if hole_pop is None:
            first = list(m_data.keys())[0]
            atom_ids = sorted(list(m_data[first].keys()))
            hole_pop = np.zeros(max(atom_ids)+1); elec_pop = np.zeros(max(atom_ids)+1)
        
        for v in range(nv):
            if (v_start + v) in m_data:
                w = np.sum(weights[k_idx, v, :])
                for atom in atom_ids:
                    d = m_data[v_start + v][atom]
                    hole_pop[atom] += w * d['total']
                    if atom not in hole_atom_orbs: hole_atom_orbs[atom] = np.zeros(MAX_ORBS)
                    orbs = d['orbitals']; n = len(orbs)
                    if n > max_orb: max_orb = n
                    if n <= MAX_ORBS: hole_atom_orbs[atom][:n] += w * orbs

        for c in range(nc):
            if (c_start + c) in m_data:
                w = np.sum(weights[k_idx, :, c])
                for atom in atom_ids:
                    d = m_data[c_start + c][atom]
                    elec_pop[atom] += w * d['total']
                    if atom not in elec_atom_orbs: elec_atom_orbs[atom] = np.zeros(MAX_ORBS)
                    orbs = d['orbitals']; n = len(orbs)
                    if n <= MAX_ORBS: elec_atom_orbs[atom][:n] += w * orbs

    if hole_pop is None: raise ValueError('No Mulliken populations were read')
    sum_h = np.sum(hole_pop); sum_e = np.sum(elec_pop)
    print(f'Captured Mulliken population: hole={sum_h:.10f}, electron={sum_e:.10f}')
    if sum_h > 1e-9: 
        hole_pop /= sum_h
        for a in hole_atom_orbs: hole_atom_orbs[a] /= sum_h
    if sum_e > 1e-9: 
        elec_pop /= sum_e
        for a in elec_atom_orbs: elec_atom_orbs[a] /= sum_e

    out_name = f"exciton_analysis_state_{state_idx}.dat"
    valid_atoms = [idx for idx in range(len(hole_pop)) if idx in atom_ids]
    orb_labels = ['s', 'p', 'd', 'f', 'g', 'h']
    
    with open(out_name, "w") as f:
        count_label = 'K points' if embedded is not None else 'Files'
        f.write(f"# Exciton Analysis State {state_idx}\n# {count_label}: {files_proc}\n\n")
        f.write("# SECTION 1: DETAILED ATOMIC ORBITAL BREAKDOWN\n")
        f.write("#" + "-"*60 + "\n")
        for atom in valid_atoms:
            f.write(f"Atom: {atom:<4d} | Hole Pop: {hole_pop[atom]:.6f} | Elec Pop: {elec_pop[atom]:.6f}\n")
            h_orbs = hole_atom_orbs.get(atom, np.zeros(MAX_ORBS))
            e_orbs = elec_atom_orbs.get(atom, np.zeros(MAX_ORBS))
            for i in range(max_orb):
                lab = f"l={i} ({orb_labels[i]})" if i < len(orb_labels) else f"l={i}"
                f.write(f"    {lab:<10} H: {h_orbs[i]:.6f}   E: {e_orbs[i]:.6f}\n")
            f.write("\n")
    print(f"Done! Saved to {out_name}")

def workflow_generate_cube_inputs():
    print("\n--- PHASE 3: Generate Cube Inputs ---")
    fname = input("Eigenvectors filename (ELSI / LibBSE NPZ): ").strip()
    if not os.path.exists(fname): raise FileNotFoundError(fname)
    
    state_idx = int(input("State Index: "))
    nv = int(input("Nv: ")); nc = int(input("Nc: ")); nk = int(input("Nk: "))
    v_start = int(input("Absolute ID First Valence Band: "))
    c_start = int(input("Absolute ID First Conduction Band: "))
    check_band_window(fname, v_start, c_start)

    weights = get_bse_weights(fname, state_idx, nk, nv, nc)

    w_hole = np.sum(weights, axis=2) 
    w_elec = np.sum(weights, axis=1) 

    thresh = float(input("Threshold % (e.g. 1.0): ")) / 100.0
    
    out_file = "cube_snippet.in"
    count = 0
    
    with open(out_file, "w") as f:
        f.write(f"# Cube Snippet | State {state_idx} | Thresh {thresh*100}%\n")
        f.write("\n# --- HOLE CUBES ---\n")
        for k in range(nk):
            for v in range(nv):
                if w_hole[k, v] >= thresh:
                    count += 1
                    band_abs = v_start + v
                    f.write(f"output cube eigenstate_density {band_abs}\n")
                    f.write(f"   cube kpoint {k+1}\n")

        f.write("\n# --- ELECTRON CUBES ---\n")
        for k in range(nk):
            for c in range(nc):
                if w_elec[k, c] >= thresh:
                    count += 1
                    band_abs = c_start + c
                    f.write(f"output cube eigenstate_density {band_abs}\n")
                    f.write(f"   cube kpoint {k+1}\n")
                    
    print(f"Success! Generated {count} cube requests in {out_file}")

def workflow_analyze_cubes():
    print("\n--- PHASE 4: Average Density (Cube Summation) ---")
    if not ASE_AVAILABLE: raise RuntimeError("ASE not installed; install requirements.txt")

    fname = input("Eigenvectors filename (ELSI / LibBSE NPZ): ").strip()
    if not os.path.exists(fname): raise FileNotFoundError(fname)
    
    state_idx = int(input("State Index: "))
    nv = int(input("Nv: ")); nc = int(input("Nc: ")); nk = int(input("Nk: "))
    v_start = int(input("Absolute ID First Valence Band: "))
    c_start = int(input("Absolute ID First Conduction Band: "))
    check_band_window(fname, v_start, c_start)
    
    print("\n[Validation]")
    ctrl_file = input("Control file (default: cube_snippet.in): ").strip()
    if not ctrl_file: ctrl_file = "cube_snippet.in"
    active_cubes = parse_control_for_cubes(ctrl_file)
    if not active_cubes:
        if input("  -> No valid requests found. Continue anyway? (y/n): ").lower() != 'y': return
        use_validation = False
    else:
        print(f"  -> Found {len(active_cubes)} active cube targets.")
        use_validation = True

    print("\n[Filename Pattern]")
    default_pat = "cube_*_eigenstate_density_{:05d}_spin_1_k_point_{:04d}.cube"
    pat = input(f"Pattern (default: {default_pat}): ").strip()
    if not pat: pat = default_pat

    print("\nReading Vectors...")
    weights = get_bse_weights(fname, state_idx, nk, nv, nc)

    reference = None
    outputs = []
    for label, marginal, first_band in [
        ('hole', weights.sum(axis=2), v_start),
        ('elec', weights.sum(axis=1), c_start),
    ]:
        total = None
        captured = 0.0
        for k, band in np.ndindex(marginal.shape):
            if use_validation and (k+1, first_band+band) not in active_cubes:
                continue
            content = load_cube_target(pat, first_band+band, k+1, reference)
            if reference is None:
                reference = content
            if total is None:
                total = np.zeros_like(content['data'])
            total += marginal[k, band] * content['data']
            captured += marginal[k, band]
        if total is None or captured <= 0:
            raise ValueError(f'No {label} density captured')
        outputs.append((label, total/captured, captured))
    # Validate both channels before writing either output.
    for label, data, captured in outputs:
        name = f'avg_{label}_state_{state_idx}.cube'
        with open(name, 'w') as stream:
            write_cube(stream, reference['atoms'], data=data, origin=reference['origin'])
        print(f'Wrote {name}; captured BSE weight={captured:.10f}')
    print('Done!')

def workflow_analyze_bz():
    print("\n--- PHASE 5: BZ & Band Analysis (Text Only) ---")
    fname = input("Eigenvectors filename (ELSI / LibBSE NPZ): ").strip()
    if not os.path.exists(fname): raise FileNotFoundError(fname)
    
    state_idx = int(input("State Index: "))
    nv = int(input("Nv: ")); nc = int(input("Nc: ")); nk = int(input("Nk: "))
    nkx = int(input("Kx: ")); nky = int(input("Ky: ")); nkz = int(input("Kz: "))
    
    weights = get_bse_weights(fname, state_idx, nk, nv, nc)

    # --- PART 1: K-POINT ANALYSIS ---
    k_weights = np.sum(weights, axis=(1, 2))
    total_norm = np.sum(k_weights)
    if total_norm > 0: 
        k_weights_norm = k_weights / total_norm
        weights_norm = weights / total_norm
    else:
        k_weights_norm = k_weights
        weights_norm = weights
    
    dom_k_idx = np.argmax(k_weights_norm)
    kx, ky, kz = get_k_coords(dom_k_idx, nkx, nky, nkz, fname)
    
    print("\n" + "="*40 + "\n       A. RECIPROCAL SPACE (K-POINTS)\n" + "="*40)
    print(f"Total Norm Captured: {total_norm:.6f}")
    print(f"Dominant K-point:    Index {dom_k_idx} ({k_weights_norm[dom_k_idx]:.2%})")
    print(f"                     Coord ({kx:.3f}, {ky:.3f}, {kz:.3f})")
    
    # --- PART 2: BAND ANALYSIS ---
    print("\n" + "="*40 + "\n       B. BAND TRANSITIONS (ORBITALS)\n" + "="*40)
    trans_matrix = np.sum(weights_norm, axis=0) # Sum over K
    
    h_band_contrib = np.sum(trans_matrix, axis=1) # Sum over conduction
    e_band_contrib = np.sum(trans_matrix, axis=0) # Sum over valence
    
    print(f"\n[Dominant Hole Bands] (Total Nv={nv})")
    print(f"{'Band ID':<10} {'Label':<12} {'Contribution':<15}")
    print("-" * 40)
    h_sort = np.argsort(h_band_contrib)[::-1]
    for i in h_sort[:5]: 
        val = h_band_contrib[i]
        if val < 0.001: continue
        diff = (nv - 1) - i
        label = "HOMO" if diff == 0 else f"HOMO-{diff}"
        print(f"{i:<10d} {label:<12} {val:.2%}")

    print(f"\n[Dominant Electron Bands] (Total Nc={nc})")
    print(f"{'Band ID':<10} {'Label':<12} {'Contribution':<15}")
    print("-" * 40)
    e_sort = np.argsort(e_band_contrib)[::-1]
    for i in e_sort[:5]: 
        val = e_band_contrib[i]
        if val < 0.001: continue
        label = "LUMO" if i == 0 else f"LUMO+{i}"
        print(f"{i:<10d} {label:<12} {val:.2%}")

    print(f"\n[Dominant Transitions (v -> c)]")
    print(f"{'Transition':<25} {'Weight':<10}")
    print("-" * 40)
    flat_indices = np.argsort(trans_matrix.flatten())[::-1]
    for idx in flat_indices[:5]:
        val = trans_matrix.flatten()[idx]
        if val < 0.001: continue
        v = idx // nc; c = idx % nc
        h_label = "HOMO" if v == nv-1 else f"HOMO-{nv-1-v}"
        e_label = "LUMO" if c == 0 else f"LUMO+{c}"
        print(f"{h_label} -> {e_label:<12} {val:.2%}")

def workflow_generate_cond_inputs():
    print("\n--- PHASE 6: Generate Conditional Wavefunction Inputs ---")
    fname = input("Eigenvectors filename (ELSI / LibBSE NPZ): ").strip()
    if not os.path.exists(fname): raise FileNotFoundError(fname)
    
    state_idx = int(input("State Index: "))
    nv = int(input("Nv: ")); nc = int(input("Nc: ")); nk = int(input("Nk: "))
    v_start = int(input("Absolute ID First Valence Band: "))
    c_start = int(input("Absolute ID First Conduction Band: "))
    check_band_window(fname, v_start, c_start)

    weights = get_bse_weights(fname, state_idx, nk, nv, nc)

    w_hole = np.sum(weights, axis=2) 
    w_elec = np.sum(weights, axis=1)
    thresh = float(input("Threshold % (e.g. 1.0): ")) / 100.0
    
    out_file = "cube_cond_snippet.in"
    count = 0
    with open(out_file, "w") as f:
        f.write(f"# Conditional Snippet | State {state_idx} | Thresh {thresh*100}%\n")
        f.write("\n# --- HOLE WAVEFUNCTIONS ---\n")
        for k in range(nk):
            for v in range(nv):
                if w_hole[k, v] >= thresh:
                    count += 1
                    band_abs = v_start + v
                    f.write(f"output cube eigenstate {band_abs}\n")
                    f.write(f"   cube kpoint {k+1}\n")
                    f.write(f"output cube eigenstate_imag {band_abs}\n   cube kpoint {k+1}\n")
        f.write("\n# --- ELECTRON WAVEFUNCTIONS ---\n")
        for k in range(nk):
            for c in range(nc):
                if w_elec[k, c] >= thresh:
                    count += 1
                    band_abs = c_start + c
                    f.write(f"output cube eigenstate {band_abs}\n")
                    f.write(f"   cube kpoint {k+1}\n")
                    f.write(f"output cube eigenstate_imag {band_abs}\n   cube kpoint {k+1}\n")
    print(f"Success! Generated {2*count} real/imaginary wavefunction requests in {out_file}")
    if nk > 1: print("Producer requirement: off-Gamma complex cubes in the SAME band gauge as the BSE amplitudes. Some FHI-aims versions disable this output.")

def workflow_analyze_cond():
    print("\n--- PHASE 7: Conditional Density Analysis ---")
    if not ASE_AVAILABLE: raise RuntimeError("ASE not installed; install requirements.txt")
    fname = input("Eigenvectors filename (ELSI / LibBSE NPZ): ").strip()
    if not os.path.exists(fname): raise FileNotFoundError(fname)
    
    state_idx = int(input("State Index: "))
    nv = int(input("Nv: ")); nc = int(input("Nc: ")); nk = int(input("Nk: "))
    v_start = int(input("Abs Valence ID: "))
    c_start = int(input("Abs Conduction ID: "))
    check_band_window(fname, v_start, c_start)

    print("\n[Fixed Hole Position]")
    hx = float(input("  x (Ang): ")); hy = float(input("  y (Ang): ")); hz = float(input("  z (Ang): "))
    r_fix = np.array([hx, hy, hz])

    print("\n[Pattern]")
    default_pat = "cube_*_eigenstate_{:05d}_spin_1_k_point_{:04d}.cube"
    pat = input(f"Pattern (default: {default_pat}): ").strip() or default_pat

    print("\nReading Wavefunction Coefficients...")
    get_bse_weights(fname, state_idx, nk, nv, nc)  # validate all dimensions
    coeffs = read_elsi_state(fname, state_idx).toarray().reshape(nk,nv,nc)
    if str(fname).endswith('.npz'):
        with np.load(fname, allow_pickle=False) as archive:
            if str(archive['wavefunction_gauge']) != 'native':
                raise ValueError('Conditional density requires native gauge or an explicit orbital gauge transformation')
    thresh = float(input("Threshold % used in Phase 6: ")) / 100.0
    if not 0 <= thresh <= 1:
        raise ValueError('Threshold must lie between 0 and 100 percent')
    if '_eigenstate_' not in pat:
        raise ValueError('Complex cube pattern must contain _eigenstate_; imaginary partner uses _eigenstate_imag_')
    imag_pat = pat.replace('_eigenstate_', '_eigenstate_imag_')
    reference = None

    def orbital(band, k):
        nonlocal reference
        real = load_cube_target(pat, band, k, reference)
        if reference is None:
            reference = real
        imag = load_cube_target(imag_pat, band, k, reference)
        return real['data'] + 1j*imag['data']

    hole_mask = np.sum(abs(coeffs)**2, axis=2) >= thresh
    elec_mask = np.sum(abs(coeffs)**2, axis=1) >= thresh
    hole_vals = np.zeros((nk,nv), dtype=complex)
    for k,v in zip(*np.where(hole_mask)):
        data = orbital(v_start+v, k+1)
        hole_vals[k,v] = get_grid_val_at_coord(data, reference['atoms'], r_fix, reference['origin'])
    if reference is None:
        raise ValueError('No hole bands selected')
    mixing = np.einsum('kvc,kv->kc', coeffs, hole_vals.conj())
    total_psi = np.zeros_like(reference['data'], dtype=complex)
    for k,c in zip(*np.where(elec_mask)):
        total_psi += mixing[k,c] * orbital(c_start+c,k+1)
    density = abs(total_psi)**2
    captured = np.sum(abs(coeffs)**2 * hole_mask[:,:,None] * elec_mask[:,None,:])
    if not np.isfinite(density).all() or not np.any(density > 0):
        raise ValueError('Conditional density is zero or non-finite at this hole position')
    name = f'cond_density_state_{state_idx}.cube'
    with open(name, 'w') as stream:
        write_cube(stream, reference['atoms'], data=density, origin=reference['origin'])
    print(f'Written {name}; retained pair weight={captured:.10f}; raw |Psi|^2 (not unit normalized)')

# =============================================================================
#                                MAIN MENU
# =============================================================================

def main():
    print("==================================================")
    print("          BSE ANALYSIS TOOLKIT (ExciView)         ")
    print("==================================================")
    print("1. Phase 1: Generate Mulliken Inputs")
    print("2. Phase 2: Analyze Mulliken Output")
    print("3. Phase 3: Generate Cube Inputs (Avg Density)")
    print("4. Phase 4: Analyze Cube Files (Avg Density)")
    print("5. Phase 5: BZ & Band Analysis (Text Only)")
    print("6. Phase 6: Generate Conditional Inputs")
    print("7. Phase 7: Analyze Conditional Density")
    print("0. Exit")
    
    choice = input("\nSelect Option: ").strip()
    
    if choice == "1": workflow_generate_inputs_mulliken()
    elif choice == "2": workflow_analyze_spatial()
    elif choice == "3": workflow_generate_cube_inputs()
    elif choice == "4": workflow_analyze_cubes()
    elif choice == "5": workflow_analyze_bz()
    elif choice == "6": workflow_generate_cond_inputs()
    elif choice == "7": workflow_analyze_cond()
    elif choice == "0": sys.exit(0)
    else: print("Invalid option.")

if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, IndexError, RuntimeError) as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        sys.exit(1)

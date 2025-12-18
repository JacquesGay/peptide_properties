import os
import glob
from collections import defaultdict
from typing import Dict, Tuple, List
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import distances, dihedrals

import nestiness_analysis_functions as nest
import traj_split
# =============================
# ----------SETTINGS-----------
# =============================
cutoff = 10.0
cutoff_list = np.arange(3.5,4.0,0.25).tolist()
repeat_number = 3
ligands = ["apo"]
pdbs = ["gpsgsgks"]
job_name = 'prod'
traj_name = f'{job_name}_full.xtc'
flush = False
GMX_COMMAND = 'gmx_mpi'
build_filtered = True
COMBINE_REPEATS = True

master_directory = os.getcwd()
job_directory = os.path.join(master_directory, job_name)

amino_acids = {
        'Pos': ['ARG', 'HIS', 'LYS'],
        'Neg': ['ASP', 'GLU'],
        'Pol': ['SER', 'THR', 'ASN', 'GLN', 'TYR', 'CYS'],
        'Nonpol': ['GLY', 'PRO', 'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP']
        }

aa_colors = {
        'Pos': 'r',
        'Neg': 'm',
        'Pol': 'g',
        'Nonpol': 'b'
        }

# ============================
# -----------HELPERS----------
# ============================

def tree():
    """Recursively-nested dict that auto-creates children"""
    return defaultdict(tree)

def find_first(path_glob: str) -> str:
    """Return the first match for a glob pattern, or raise error"""
    matches = glob.glob(path_glob)
    if not matches:
        raise FileNotFoundError(f"[ERROR] No files matched: {path_glob}")
    return matches[0]

def create_universe(
        job_directory: str,
        ligand_name: str,
        pdb: str,
        repeat: str,
        traj: str,
        flush: bool = flush,
        ) -> mda.Universe:
    """
    Build an MDAnalysis Universe for one replica.
    Resolves the working directory structure and finds the first *.gro
    """

    rep_dir = f"REP{repeat}_flush" if flush else f"REP{repeat}"
    work_directory = os.path.join(job_directory, ligand_name, pdb, rep_dir)
    gro_path = None
    try:
        gro_path = find_first(os.path.join(work_directory, "*.gro"))
    except FileNotFoundError as e:
        print(f'{e}: Using npt')

    if gro_path is None:
        print("Using npt gro")
        npt_dir = os.path.join(master_directory, 'npt', ligand_name, pdb, rep_dir)
        matches = glob.glob(os.path.join(npt_dir, "*.gro"))
        gro_path = matches[0]
    return mda.Universe(gro_path, traj) #temp changed from traj_path for string of trajectories

def residue_labels_from_ag(ag: mda.core.groups.AtomGroup) -> Tuple[List[int], List[str]]:
    """
    Produce residue interger IDs and labels like 'GLY1' in residue order.
    Ensures unique labels per residue
    """
    resids = ag.resids.tolist()
    labels = []
    seen = set()
    for atom in ag:
        label = f"{atom.resname}{atom.resid}"
        if label not in seen:
            labels.append(label)
            seen.add(label)
    return resids, labels

def classify_residue_color(res_label: str) -> str:
    """
    Map a residue label to a color based on its amino acid properties
    """
    # Extract 3-letter code from label
    resname = ''.join(ch for ch in res_label if not ch.isdigit())
    for grp, names in amino_acids.items():
        if resname in names:
            return aa_colors[grp]

def df_statistics(
        df: pd.DataFrame,
        total_contacts: bool = True,
        samples: int = 0
        ) -> pd.DataFrame:
    """
    Compute per-column summary statistics with 95% CI. If samples==0, use len(df).
    NaN values are ignored.
    """
    if samples == 0:
        samples = len(df)
    numeric = df.select_dtypes(include=[np.number]).copy()
    stat_df = pd.DataFrame({
        "avg": numeric.mean(axis=0, skipna=True),
        "median": numeric.median(axis=0, skipna=True),
        "std": numeric.std(axis=0, skipna=True)
        })
    stat_df["ci"] = 1.96 * (stat_df["std"] / np.sqrt(max(samples,1)))
    if total_contacts:
        stat_df["total_contacts"] = numeric.sum(axis=0, skipna=True)
    return stat_df

def bar_with_ci(
        stats_df: pd.DataFrame,
        title: str,
        ylabel: str,
        outfile: str,
        add_ci: bool = True,
        extra_title_note: str = "",
        ):
    """
    Draw bar plot with CI (optional). Colors bars based on amino acid class
    """
    x = stats_df.index.tolist()
    y = stats_df["avg"].to_numpy()
    yerr = stats_df["ci"].to_numpy() if add_ci else None

    # Map labels to colors
    colors = [classify_residue_color(lbl) for lbl in x]

    plt.figure(figsize=(6.4,4.8))
    plt.bar(x, y, yerr=yerr, ecolor='gray' if add_ci else None, color=colors)
    full_title = title if not extra_title_note else f"{title}; {extra_title_note}"
    plt.title(full_title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    os.makedirs("analysis", exist_ok=True)
    plt.savefig(os.path.join(f"analysis/{outfile}"), dpi=300)
    plt.close()

def write_frames_ndx(path, frames):
    """write a [ frames ] group with 0-based frame numbers for gmx trjconv"""
    with open(path, "w") as fh:
        fh.write("[ frames ]\n")
        line = []
        for i, fr in enumerate(sorted(int(f) for f in frames)):
                line.append(str(fr))
                if (i+1) % 15 == 0:
                    fh.write(" ".join(line) + "\n")
                    line = []
        if line:
            fh.write(" ".join(line) + "\n")

def trjconv_filter_frames(gmx_command, s_ref, xtc_in, frames, xtc_out, work_dir):
    """
    create filtered trajectory
    """
    frames_ndx = os.path.join(work_dir, "frames.ndx")
    write_frames_ndx(frames_ndx, frames)
    cmd = [gmx_command, "trjconv",
        "-s", s_ref,
        "-f", xtc_in,
        "-fr", frames_ndx,
        "-o", xtc_out]
    subprocess.run(cmd, input="0\n",check=True, text=True)#, capture_output=True)

def trjcat_concat(gmx_command, xtc_list, xtc_out):
    """Concatenate multiple XTCs into one"""
    cmd = [gmx_command, "trjcat", "-cat", "-keeplast", "-f",]
    for x in xtc_list:
        cmd += [ x ]
    cmd += ["-o", xtc_out]
    subprocess.run(cmd, check=True, text=True)#, capture_output=True)
    return xtc_out

def per_rep_frame_slices(global_frames, rep_frame_counts):
    """
    Convert global frame numbers (1000 total frames in 3 repeats) into per-Rep frame indices (frame 200 in REP2)
    """
    offsets, per_rep = [], [[] for _ in rep_frame_counts]
    s = 0
    for n in rep_frame_counts:
        offsets.append(s); s += n
    for fr in global_frames:
        for r, off in enumerate(offsets):
            if off <= fr < off + rep_name_counts[r]:
                per_rep[r].append(fr - off); break
    return per_rep

# ============================
# ----CORE FRAME ANALYSIS-----
# ============================

def distances_per_frame(
        u: mda.Universe,
        cutoff: float,
        surface_sel: str = "resname LAP", # and not name OB",
        peptide_sel: str = "protein and not name H*",
        dtype=np.float32,
        contact_conditions=True
        ) -> Tuple[Dict[str, Dict[str, np.ndarray]], mda.core.groups.AtomGroup, mda.core.groups.AtomGroup]:
    """
    For each frame:
        - Build a (n_surface_top x n_peptide_heavy) distance matrix with NaN fill
        - Compute a binary contact mask per peptide atom (1 if any surface atom within cutoff)
        - Keep the minimum distance per peptide atom (NaN if no contact that frame)

    Returns:
        time_dict: { "time_ns_str": {"min_per_col": 1D array, "mask_per_col": 1D Array, "M": 2DArray}
        }
        top_layer_ag, peptide_ag
    """
    # Atom selections (top-layer determined from initial frame)
    surface = u.select_atoms(surface_sel)
    #top_layer = surface.select_atoms(f"prop z > {surface.positions[:, 2].max() - 1.0}")
    Lz = float(u.dimensions[2])
    layer_thickness = 1.0
    z = surface.positions[:, 2].astype(float)

    order = np.argsort(z)
    zs = z[order]
    diffs = np.diff(zs, append=zs[0] + Lz)
    imax = np.argmax(diffs)

    z_top_of_bottom = zs[imax]
    dz_centered = ((z - z_top_of_bottom + 0.5*Lz) % Lz) -0.5*Lz
    crest_narrow = surface[np.abs(dz_centered) < 0.2]
    if len(crest_narrow) == 0:
        seed = surface[np.argmin(np.abs(dz_centered))]
    else:
        seed = crest_narrow[0]

    target_name = seed.name
    same_kind = (np.array(surface.names) == target_name)
    by_height = (np.abs(dz_centered) <= layer_thickness)
    top_layer = surface[same_kind & by_height]

    peptide = u.select_atoms(peptide_sel)
    if contact_conditions:
        contact_residues = traj_split.set_residues_marking_adsorption(peptide)
        print(contact_residues)

    top_layer.write("top_layer.pdb")
    u.select_atoms("all").write("universe.pdb")
    n, m = top_layer.n_atoms, peptide.n_atoms
    time_dict = {}
    contact_frames = []

    # Iterate trajectory once; minimize Python per-frame overhead
    for ts in u.trajectory:
        # Initialize distance matrix with NaN (fast fill)
        M = np.full((n,m), np.nan, dtype=dtype)
        minimized_array = np.full((n,m), np.nan, dtype=dtype)

        # Compute all pairs within cutoff
        # pairs: (k,2) with rows = surface idx, cols = peptide idx (0-based)
        pairs, dists = distances.capped_distance(
                top_layer, peptide, max_cutoff=cutoff, return_distances=True
                )
        
        if contact_conditions:
            contacted = set(pairs[:, 1]) if pairs.size else set()
            
            if len(pairs) >= 2 and any(res in contacted for res in contact_residues): 
            # set(contact_residues).issubset(contacted):
            #if len(pairs) >=1 or any(res in contacted for res in contact_residues): #make on contact out of the options
                #if pairs.size:
                rows = pairs[:, 0]
                cols = pairs[:, 1]
                M[rows, cols] = dists

                #if np.any(~np.isnan(M)):
                contact_frames.append(ts.frame)
        
        else:
            if pairs.size:
                #vectorized insert of distances into M
                rows = pairs[:, 0]
                cols = pairs[:, 1]
                M[rows, cols] = dists
        
            if np.any(~np.isnan(M)):
                contact_frames.append(ts.frame)
        # Keep only minimum distance from each atom in the timestep
        M_inf = M.copy()
        M_inf[np.isnan(M_inf)] = np.inf
        min_row = np.min(M_inf, axis =0)
        min_row[np.isinf(min_row)] = np.nan

        masked_peptide = (~np.isnan(min_row)).astype(np.int8)
        # Binary mask: 1 if contact exists, 0 otherwise
        #save time in ns
        time_ns = (ts.frame + 1) * 1.25 #Convert to time value by using the frame of the trajectory and multiply by the save value from mdp file
        time_key = f"{time_ns:.5e}"

        time_dict[time_key] = {
                "M": M,
                "min_per_col": min_row,
                "mask_per_col":masked_peptide,
                }

    return time_dict, top_layer, peptide, contact_frames

# ============================
# ------CUTOFF TRAJECTORY-----
# ============================
def build_filtered_xtc_for_system(
        gmx_command,
        job_directory,
        ligand,
        pdb,
        repeat_number,
        traj_name,
        cutoff,
        per_rep_contact_frames):
    pdb_dir = os.path.join(job_directory, ligand, pdb)
    combined_out = os.path.join(pdb_dir, f"adsorbed_c{cutoff}.xtc")
    filtered_xtcs = []

    for repeat in range(repeat_number):
        frames = per_rep_contact_frames.get(repeat, [])
        if not frames:
            continue
        rep_dir = os.path.join(pdb_dir, f"REP{repeat}")
        s_ref_tpr = os.path.join(rep_dir, f"{job_name}.tpr")
        try:
            gro_name = glob.glob(os.path.join(rep_dir, "*.gro"))[0]
        except IndexError as e:
            #raise FileNotFoundError(f"No .gro file found in {rep_dir}")
            print(f"No .gro file found in {rep_dir}")
            gro_name = glob.glob(os.path.join(os.getcwd(), 'npt', ligand, pdb, f"REP{repeat}", "*.gro"))[0]
        s_ref = s_ref_tpr if os.path.exists(s_ref_tpr) else gro_name

        xtc_in = os.path.join(rep_dir, traj_name)
        xtc_out = os.path.join(rep_dir, f"adsorbed_c{cutoff}.xtc")

        trjconv_filter_frames(
                gmx_command = gmx_command,
                s_ref = s_ref,
                xtc_in = xtc_in,
                frames = frames,
                xtc_out = xtc_out,
                work_dir = rep_dir
                )
        filtered_xtcs.append(xtc_out)
    if filtered_xtcs:
        trjcat_concat(gmx_command, filtered_xtcs, combined_out)
        print(f"[OK] Built {combined_out}")
        return combined_out, True
    else:
        print(f"[ERROR] No adsorbed frames for {pdb}:{ligand} cutoff={cutoff}")
        return None, False

# ============================
# ----------ANALYSIS----------
# ============================


def summarize_mask_over_repeats(
        data_dict, cutoff: float, ligand: str, pdb: str, pep_ag, #repeat_num: int
        ):
    """
    for each repeat:
        - sum per-frame mask vectors (contacts per residue per frame) across frames
    Then:
        - Build a per-repeat summary table and a per-timestep table (all frames concatenated)
        - Plot: (1) mean contacts per timestep +/- CI, (2) total contacts per repeat
    """

    resids, labels = residue_labels_from_ag(pep_ag)

    per_repeat_sums = []
    all_frame_rows = []

    for repeat in range(repeat_number):
        # data for a single repeat: dict keyed by time
        tdict = data_dict[cutoff][ligand][pdb][repeat]
        # Gather the repeats frame masks
        frame_masks = [tdict[t]["mask_per_col"] for t in sorted(tdict.keys())]
        frame_masks = np.vstack(frame_masks) if frame_masks else np.empty((0, len(resids)))
        # Sum across all frames for this repeat
        per_repeat_sums.append(frame_masks.sum(axis=0))
        # Also keep per-frame rows for per-timestep stats
        all_frame_rows.append(frame_masks)

    # Combine
    per_repeat_sums = np.vstack(per_repeat_sums) if per_repeat_sums else np.empty((0, len(resids)))
    all_frame_rows = np.vstack(all_frame_rows) if all_frame_rows else np.empty((0, len(resids)))

    # Make DataFrames with residue grouping (columns may repeat across atom-group by resid to sum)
    df_rep = pd.DataFrame(per_repeat_sums, columns=resids).groupby(level=0, axis=1).sum()
    df_rep.columns = labels
    df_ts = pd.DataFrame(all_frame_rows, columns=resids).groupby(level=0, axis=1).sum()
    df_ts.columns = labels

    # Stats
    ts_stats = df_statistics(df_ts, total_contacts=False)
    whole_stats = df_statistics(df_rep, total_contacts=True)

    #Plots
    bar_with_ci(
            ts_stats,
            title="Contacts per timestep",
            ylabel="Contacts (+/-% CI)",
            outfile=f'masked_contacts_{cutoff}.png',
            add_ci=True
            )
    bar_with_ci(
            whole_stats,
            title="Contacts per repeat",
            ylabel="Contacts",
            outfile=f"masked_contacts_total_{cutoff}.png",
            add_ci=False,
            extra_title_note=f"Total {whole_stats['total_contacts'].sum()}"
            )

def summarize_min_dists_over_repeats(
        data_dict, cutoff: float, ligand: str, pdb: str, pep_ag, #repeat_num: int
        ):
    """
    Aggregate per-frame minimum distances (per peptide atom) across repeats, then
    report the per-residue average min distance +/- CI (NaN columns dropped)
    """
    resids, labels = residue_labels_from_ag(pep_ag)

    #Collect all minimized rows from all repeats
    rows = []
    for repeat in range(repeat_number):
        tdict = data_dict[cutoff][ligand][pdb][repeat]
        rows.extend([tdict[t]["min_per_col"] for t in sorted(tdict.keys())])

    if not rows:
        return

    #arr = np.stack(rows) # shape (num_frames, num_pep_atoms)
    # Build DF and group columns by resids and reduce by the minimum per risidue
    df = pd.DataFrame(rows, columns=resids)
    df = df.groupby(df.columns, axis=1).min()
    df = df.dropna(how='all') #drop all NaN values
    df.columns = labels

    stats_df = df_statistics(df, total_contacts=False)

    bar_with_ci(
            stats_df,
            title=f"Average minimum distance per residue (cutoff {cutoff})",
            ylabel=r"Distance ($\AA$) +/- 95% CI",
            outfile=f'min_distance_{cutoff}.png',
            add_ci=True
            )

def distance_histogram(
        data_dict, cutoff: float, ligand: str, pdb: str, pep_ag,
        ):
    """
    Construct histogram of minimum distance of peptide to surface in cutoff distance over trajectory
    """
    resids, labels = residue_labels_from_ag(pep_ag)

    rows = []
    for repeat in range(repeat_number):
        tdict = data_dict[cutoff][ligand][pdb][repeat]
        rows.extend([tdict[t]["min_per_col"] for t in sorted(tdict.keys())])
    
    if not rows:
        return

    df = pd.DataFrame(rows, columns=resids)
    df = df.groupby(df.columns, axis=1).min()
    df = df.dropna(how='all')
    df.columns = labels

    bins = np.arange(2.50, 5.25, 0.25)
    
    axes = df.hist(bins = bins, figsize = (6.4,4.8), sharex = True, sharey = True, grid=False, density=True)

    plt.suptitle(f"Min peptide-surface distance per residue")
    for ax, col in zip(np.ravel(axes), df.columns):
        if ax is not None:
            ax.set_title(col)
            ax.set_xlabel("Distance (A)")
            ax.set_ylabel("Counts")

    plt.tight_layout()
    plt.show()

def rama_make(u, pdb, ligand):
    protein = u.select_atoms("protein")
    rama = dihedrals.Ramachandran(protein)
    rama.run()

    fig, ax = plt.subplots(figsize=plt.figaspect(1))
    rama.plot(ax=ax, marker='s')
    # plt.show()
    plt.savefig(f'analysis/rama_{pdb}_{ligand}.png', dpi = 300)
    del rama
    del fig
    del ax

   
def rama_plot(rama):
    fig, ax = plt.subplots(figsize=plt.figaspect(1))
    rama.plot(color='blue', marker='.')
    plt.show()

# ==============================
# ----------- MAIN ------------
# ==============================


def main_combined():
    cutoff_dict = tree()
    
    for cutoff in cutoff_list:
        for ligand in ligands:
            for pdb in pdbs:
                #build per-repeat dictionaries
                pep_ag_ref = None
                per_rep_contact_frames = {}
                
                for repeat in range(repeat_number):
                    xtc_path = os.path.join(job_directory, ligand, pdb, f"REP{repeat}", traj_name)
                    u = create_universe(job_directory, ligand, pdb, repeat, xtc_path)
                    tdict, top_ag, pep_ag, contact_frames = distances_per_frame(u, cutoff=cutoff,contact_conditions=True)
                    cutoff_dict[cutoff][ligand][pdb][repeat] = tdict
                    per_rep_contact_frames[repeat] = contact_frames
                    if pep_ag_ref is None:
                        pep_ag_ref = pep_ag 

                summarize_min_dists_over_repeats(cutoff_dict, cutoff, ligand, pdb, pep_ag_ref)
                summarize_mask_over_repeats(cutoff_dict, cutoff, ligand, pdb, pep_ag_ref)
                try:
                    gro_name = glob.glob(os.path.join(job_directory, ligand, pdb, f"REP0", "*.gro"))[0].split('.gro')[0]
                except IndexError as e:
                    print(f"using npt gro")
                    gro_name = None
                if gro_name is None:
                    gro_name = glob.glob(os.path.join(master_directory, 'npt', ligand, pdb, f"REP0", "*.gro"))[0].split('.gro')[0]
                if build_filtered:
                    print("building combined adsorbed XTC")
                    print(per_rep_contact_frames)
                    print(len(per_rep_contact_frames[2]))
                    combined_xtc, filtered = build_filtered_xtc_for_system(
                            gmx_command = GMX_COMMAND,
                            job_directory = job_directory,
                            ligand = ligand,
                            pdb = pdb,
                            repeat_number=repeat_number,
                            traj_name = traj_name,
                            cutoff = cutoff,
                            per_rep_contact_frames = per_rep_contact_frames
                            )
                    print("Combined XTC written")
                
                if filtered:
                    try:
                        print("Attempting to build nest figure")
                        nest.dihedral_nestiness_figure(
                            pdb = pdb,
                            ligand = ligand,
                            master_dir = master_directory,
                            #res1=1,#res1=4,
                            #res2=7,
                            prev_gro_name = gro_name,
                            job_name = job_name,
                            number_repeats=repeat_number,
                            use_cutoff_xtc=bool(combined_xtc),
                            cutoff=cutoff if combined_xtc else None,
                            out_png=f"nest_classify_{pdb}_{ligand}_c{cutoff}.png"
                            )
                    except Exception as e:
                        print(f"[WARN] nestiness plotting failed for {pdb}:{ligand} c={cutoff}: {e}")

                    try:
                        nest.rog_run(
                            pdb,
                            ligand,
                            master_directory,
                            cutoff=cutoff,
                            number_repeats=repeat_number,
                            job_name=job_name,
                            traj_name=None,
                            out_png=f"ROG_classify_{pdb}_{ligand}.png",
                            surface =True,)
                    except Exception as e:
                        print(f"[Warn] ROG plotting failed for {pdb}:{ligand} c={cutoff}: {e}")
                    
                    if combined_xtc:
                        try:
                            df_dist, df_angle = nest.dist_and_angle_run(
                                pdb=pdb,
                                ligand=ligand,
                                master_dir=master_directory,
                                cutoff = cutoff,
                                number_repeats=repeat_number,
                                traj_name=traj_name,
                                gmx_command = GMX_COMMAND,
                                job_name=job_name,
                                out_png=f"{pdb}_{ligand}_c{cutoff}.png",
                                analysis_dir=None,
                                do_plots=True,
                                traj_override=combined_xtc,
                                combine_repeats = COMBINE_REPEATS)
                        except Exception as e:
                            print(f"[WARN] distance/angle plotting failed for {pdb}:{ligand} c={cutoff}: {e}")
                        try:
                            nest.process_abego(
                                    master_directory,
                                    pdb,
                                    ligand,
                                    job_name,
                                    prev_gro_name = gro_name,
                                    number_repeats=repeat_number,
                                    traj_name = None,
                                    cutoff = cutoff,
                                    indexing_pairs = [(1,8)],
                                    labels=None,
                                    traj_override=combined_xtc,
                                    )

                            print(f"[SUCCESS] ABEGO plot made")
                        except Exception as e:
                            print(f"[WARN] ABEGO plotting failed for {pdb}:{ligand} c ={cutoff}: {e}")
                        try:
                            print(f"[TRY] making RMSD")
                            nest.rmsd(
                                ligand_name=ligand,
                                pdb =pdb,
                                master_directory= master_directory,
                                job_name = job_name,
                                number_of_repeats=repeat_number,
                                traj_override=combined_xtc,
                                cutoff=cutoff)
                            print(f"[SUCCESS] RMSD plot made")
                        except Exception as e:
                            print(f"[WARN] RMSD plotting failed")
                        try:
                            print(f"[TRY] contacts on adsorbed frames")
                            nest.contacts_run(
                                pdb=pdb,
                                ligand=ligand,
                                master_dir=master_directory,
                                job_name=job_name,
                                number_repeats=repeat_number,
                                traj_name=traj_name,
                                gmx_command=GMX_COMMAND,
                                prev_gro_name=gro_name,
                                prev_gro_dir="npt",
                                traj_override=combined_xtc,   # adsorbed traj
                                combine_repeats=True,
                                cutoff=cutoff,
                                    )
                            print(f"[SUCCESS] contacts on adsorbed frames")
                        except Exception as e:
                            print(f"[WARN] contacts on adsorbed frames failed for "
                              f"{pdb}:{ligand} c={cutoff}: {e}")

def main_per_repeat():
    cutoff_dict = tree()

    for cutoff in cutoff_list:
        for ligand in ligands:
            for pdb in pdbs:
                pep_ag_ref = None
                per_rep_contact_frames = {}

                # --- gather contact frames per repeat ---
                for repeat in range(repeat_number):
                    xtc_path = os.path.join(job_directory, ligand, pdb, f"REP{repeat}", traj_name)
                    u = create_universe(job_directory, ligand, pdb, repeat, xtc_path)
                    tdict, top_ag, pep_ag, contact_frames = distances_per_frame(
                        u, cutoff=cutoff, contact_conditions=True
                    )
                    cutoff_dict[cutoff][ligand][pdb][repeat] = tdict
                    per_rep_contact_frames[repeat] = contact_frames
                    if pep_ag_ref is None:
                        pep_ag_ref = pep_ag

                # --- summaries (across repeats) still OK to aggregate if you want ---
                summarize_min_dists_over_repeats(cutoff_dict, cutoff, ligand, pdb, pep_ag_ref)
                summarize_mask_over_repeats(cutoff_dict, cutoff, ligand, pdb, pep_ag_ref)

                gro_name = glob.glob(os.path.join(job_directory, ligand, pdb, "REP0", "*.gro"))[0].split(".gro")[0]

                # Build per-rep filtered XTCs; we’ll *use* the per-rep list for plotting
                combined_xtc = None
                filtered = False
                if build_filtered:
                    combined_xtc, filtered = build_filtered_xtc_for_system(
                        gmx_command=GMX_COMMAND,
                        job_directory=job_directory,
                        ligand=ligand,
                        pdb=pdb,
                        repeat_number=repeat_number,
                        traj_name=traj_name,
                        cutoff=cutoff,
                        per_rep_contact_frames=per_rep_contact_frames,
                    )

                # Collect the per-rep filtered files (if any)
                per_rep_xtcs = []
                for r in range(repeat_number):
                    rep_xtc = os.path.join(job_directory, ligand, pdb, f"REP{r}", f"adsorbed_c{cutoff}.xtc")
                    if os.path.exists(rep_xtc):
                        per_rep_xtcs.append(rep_xtc)
                if not per_rep_xtcs:
                    # If no filtered per-rep XTCs exist, fall back to the original full trajectories
                    per_rep_xtcs = [
                        os.path.join(job_directory, ligand, pdb, f"REP{r}", traj_name)
                        for r in range(repeat_number)
                    ]

                # --- nestiness (per-repeat) ---
                # Your current dihedral_nestiness_figure doesn’t have combine_repeats,
                # so we call it once per repeat by passing xtc_override and a suffix.
                for r, xtc in enumerate(per_rep_xtcs):
                    try:
                        nest.dihedral_nestiness_figure(
                            pdb=pdb,
                            ligand=ligand,
                            master_dir=master_directory,
                            prev_gro_name=gro_name,
                            job_name=job_name,
                            number_repeats=repeat_number,
                            use_cutoff_xtc=False,      # we’re overriding xtc explicitly
                            cutoff=None,
                            xtc_override=xtc,          # <— per-repeat traj
                            out_png=f"nest_classify_{pdb}_{ligand}_c{cutoff}_rep{r}.png",
                        )
                    except Exception as e:
                        print(f"[WARN] nestiness per-repeat failed for {pdb}:{ligand} REP{r} c={cutoff}: {e}")

                # --- distance/angle (per-repeat plots) ---
                try:
                    nest.dist_and_angle_run(
                        pdb=pdb,
                        ligand=ligand,
                        master_dir=master_directory,
                        cutoff=cutoff,
                        number_repeats=repeat_number,
                        traj_name=traj_name,
                        gmx_command=GMX_COMMAND,
                        job_name=job_name,
                        out_png=f"{pdb}_{ligand}_c{cutoff}.png",
                        analysis_dir=None,
                        do_plots=True,
                        # Per-repeat mode: pass the list and let the function plot REP0/1/2 separately
                        traj_override=per_rep_xtcs,
                        combine_repeats=False,
                    )
                except Exception as e:
                    print(f"[WARN] distance/angle per-repeat failed for {pdb}:{ligand} c={cutoff}: {e}")


def old_main():
    cutoff_dict = tree()
    for cutoff in cutoff_list:
        for ligand in ligands:
            for pdb in pdbs:
                pep_ag_ref = None
                trajectories = []
                r = None
                print(r)
                for repeat in range(repeat_number):
                    traj = os.path.join(job_directory, ligand, pdb, f'REP{repeat}', traj_name)
                    print(traj)
                    trajectories.append(os.path.join(job_directory, ligand, pdb, f'REP{repeat}', traj_name))
                u = create_universe(job_directory, ligand, pdb, repeat, trajectories)
                rama_make(u,pdb,ligand)
            #rama_plot(rama)

            #    tdict, top_ag, pep_ag = distances_per_frame(u, cutoff=cutoff)
            #    cutoff_dict[cutoff][ligand][pdb][repeat] = tdict
            #    for k, v in cutoff_dict.items():
            #        print(k)
            #    if pep_ag_ref is None:
            #        pep_ag_ref = pep_ag
            #distance_histogram(cutoff_dict, cutoff, ligand, pdb, pep_ag)

if __name__ == "__main__":
    main_combined()
    #main_per_repeat()

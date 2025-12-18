from __future__ import annotations

import os
import shutil
import subprocess
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
from hb_groups import * 
### ------Dynamic Variables----- ###
LIGANDS = ['apo', 'hpo4', 'gtp']
PDBS = ['gkpnvgks', 'gpessgkt', 'kgsgnkvp', 'tgpssekg']
PREV_GRO_NAME = 'prod'
NUMBER_REPEATS = 3
TRAJ_NAME = 'prod_full.xtc'
GMX_COMMAND = 'gmx_mpi'
JOB_NAME = 'prod'
PREV_GRO_DIR = 'npt'

### ------Directory Work----- ###
master_directory = os.getcwd()
analysis_directory = os.path.join(master_directory, 'analysis')
os.makedirs(analysis_directory, exist_ok=True)

#-----------------------------------------------
hb_aa_b = aa_b
hb_aa_s = aa_s
hb_aa_abbr = aa_abbr
hb_lig = lig
#################################
#----------PATHS-----------------
#################################

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def proj_paths(master_dir: str = None):
    master_directory = master_dir or os.getcwd()

    analysis_directory = os.path.join(master_directory, "analysis")
    ensure_dir(analysis_directory)

    return master_directory, analysis_directory

#################################
#---------HELPERS----------------
#################################

def run_subprocess(cmd, stdin, check: bool = True) -> subprocess.CompletedProcess:
    """
    Run a subprocess with optional stdin; capture stdout/err
    """
    proc = subprocess.run(
            cmd,
            input=stdin,
            text=True,
            capture_output=True
            )
    if check and proc.returncode != 0:
        raise RuntimeError(
                f"Command failed: {' '.join(cmd)}\n --- STDOUT ---\n{proc.stdout}\n--- STDERR ---\n{proc.stderr}")

    return proc

def coerce_universe(gro=None, trajectories=None, u=None):
    if u is not None:
        return u
    if gro is None or trajectories is None:
        raise ValueError("Provide either u=Universe or both gro and traj files")
    return mda.Universe(gro, trajectories)

def get_ligand_string(ligand_name):
    if ligand_name == 'apo':
        return 'Unliganded'
    return ligand_name.capitalize()

def get_reference_universe(master_directory, ligand_name, pdb):
    reference_structure_file = os.path.join(
            master_directory,
            'setup',
            ligand_name,
            pdb,
            'solv_ions.gro')
    return mda.Universe(reference_structure_file)

def find_topology_file(
        master_directory,
        job_name,
        ligand_name,
        pdb,
        repeat):
    directory = os.path.join(master_directory, job_name, ligand_name, pdb, f"REP{repeat}")
    gro_matches = glob.glob(os.path.join(directory, "*.gro"))

    if not gro_matches:
        print(f"[WARN] No .gro topology found in {directory}")
        npt_dir = os.path.join(master_directory, 'npt', ligand_name, pdb, f"REP{repeat}")
        gro_matches = glob.glob(os.path.join(npt_dir, "*.gro"))
        if not gro_matches:
            raise FileNotFoundError(
                f"[ERROR] No .gro file found for REP{repeat} in either {directory} or {npt_dir}"
            )
    return gro_matches[0]


def get_trajectory_path(
    master_directory: str,
    job_name: str,
    ligand_name: str,
    pdb: str,
    repeat: int,
    traj_override: str | None = None,
) -> str:
    """
    Get trajectory path for a given replica.

    Parameters
    ----------
    traj_override : str or None
        - If None: use default path
            master_directory/job_name/ligand_name/pdb/REP{repeat}/{job_name}_full.xtc
        - If a string containing '{repeat}': treated as a format pattern and formatted with repeat.
        - If a string without '{repeat}':
            - If it is an absolute path, used as-is.
            - Else it is joined to the default directory for that repeat.

    This is where you can easily plug in adsorption analysis trajectories by
    passing an override pattern that points to your adsorption xtc files.
    """
    default_directory = os.path.join(master_directory, job_name, ligand_name, pdb, f"REP{repeat}")
    default_traj = os.path.join(default_directory, f"{job_name}_full.xtc")

    if traj_override is None:
        return default_traj

    # Pattern with {repeat} -> format it
    if "{repeat}" in traj_override:
        return traj_override.format(repeat=repeat)

    # Absolute path override
    if os.path.isabs(traj_override):
        return traj_override

    # Relative override -> join with default directory
    return os.path.join(default_directory, traj_override)


#################################
#------DISTANCE AND ANGLES-------
#################################

#--------------ndx file creation and manipulation

def rename_group_best_effort(  #TODO: make more robust. 1 should not be a fall back because the new group is always the newest
        gmx_command: str,
        gro_file: str,
        ndx_file: str,
        target_name: str,
        signature = None
        ):
    """
    Try to find group number in ndx file for the signature
    If not found fall back to group 1
    """
    group_number = None
    try:
        with open(ndx_file, "r") as f:
            lines = f.readlines()
        header_lines_idx = [line for line in lines if line.startswith("[ ")]
        if signature:
            for idx, line in enumerate(header_lines_idx):
                if signature in line:
                    group_number = idx
                    if target_name == "Angles":  ##THIS PORTION HAS TO BE FIXED
                        cmd_tmp = [gmx_command, "make_ndx", "-f", gro_file, "-n", ndx_file, "-o", ndx_file]
                        group_number = group_number + 3
                        group_number_2 = "|".join([str(idx),str(idx+1)])
                        subprocess.run(cmd_tmp,
                                input=f'{group_number_2}\n1\nq\n',   #This part is critical to combining the groups that make up the angles 
                                capture_output = True,
                                text=True)
                    break
    except Exception:
        pass

    if group_number is None:
            raise TypeError(
                    f"The signature {signature} does not exist in the ndx files."
                    "please double check your signature and that the ndx file was created correctly"
                    )

    cmd = [gmx_command, "make_ndx", "-f", gro_file, "-n", ndx_file, "-o", ndx_file, "-nobackup"]
    run_subprocess(cmd, stdin=f"name {group_number} {target_name}\nq\n", check=False)

def make_ndx_for_pair(
        pdb: str,
        ligand: str,
        gmx_command: str,
        number_repeats: int,
        selection_dist: str,
        selection_angle_seq,
        selection_string_signature,
        job_name
        ) -> None:
    gro_file = os.path.join(job_name, ligand, pdb, "REP0", f"{job_name}.gro")

    if not os.path.exists(gro_file):
        gro_file = glob.glob(os.path.join(job_name, ligand, pdb, "REP0", "*.gro"))[0]
    for repeat in range(number_repeats):
        rep_dir = os.path.join(job_name, ligand, pdb, f"REP{repeat}")
        dist_ndx = os.path.join(rep_dir, "dist.ndx")
        angle_ndx = os.path.join(rep_dir, "angle.ndx")

        # ----Distance Indexing ---
        cmd = [gmx_command, "make_ndx", "-f", gro_file, "-o", dist_ndx, "-nobackup"]
        run_subprocess(cmd, stdin=f"{selection_dist}\n1\nq\n", check=True)

        rename_group_best_effort(
                gmx_command, gro_file, dist_ndx, target_name="Distance", signature=selection_string_signature
                )
        #---Angle indexing ----
        cmd2 = [gmx_command, "make_ndx", "-f", gro_file, "-o", angle_ndx, "-nobackup"]
        stdin_angle = "".join(line + "\n" for line in selection_angle_seq) + "1\nq\n"
        run_subprocess(cmd2, stdin=stdin_angle, check=True)

        rename_group_best_effort(
                gmx_command, gro_file, angle_ndx, target_name="Angles", signature=selection_string_signature
                )

def make_ndx_for_pair_in_dir(
        gro_file,
        out_dir,
        gmx_command,
        selection_dist,
        selection_angle_seq,
        selection_string_signature):
    dist_ndx = os.path.join(out_dir, "dist.ndx")
    angle_ndx = os.path.join(out_dir, "angle.ndx")

    cmd = [gmx_command, "make_ndx", "-f", gro_file, "-o", dist_ndx, "-nobackup"]
    run_subprocess(cmd, stdin=f"{selection_dist}\n1\nq\n", check=True)
    rename_group_best_effort(gmx_command, gro_file, dist_ndx, target_name="Distance",
                             signature=selection_string_signature)

    cmd2 = [gmx_command, "make_ndx", "-f", gro_file, "-o", angle_ndx, "-nobackup"]
    stdin_angle = "".join(line + "\n" for line in selection_angle_seq) + "1\nq\n"
    run_subprocess(cmd2, stdin=stdin_angle, check=True)

    rename_group_best_effort(
            gmx_command, gro_file, angle_ndx, target_name="Angles", signature=selection_string_signature
            )

    return dist_ndx, angle_ndx

#---------running the measure calculation with gmx

def run_gmx_measure(
    pdb,
    ligand,
    gmx_command,
    number_repeats,
    traj_name,
    job_name,
    measure,
    out_dir,
    master_dir,
    cutoff = None,
    traj_override = None,
    ndx_dir = None,
    combine_repeats = True
    ):
    """
    Run 'gmx distance' or 'gmx angle' for each repeat, concatenate all repeats, save one combined XVG
    Returns the combined file path
    """
    if measure not in ("dist", "angle"):
        raise ValueError("measure must be 'dist' or 'angle'")

    if measure == "dist":
        gmx_feed = "distance"
        process_input = "Distance"
        output_flag = "-oall"
        ndx_file_name = "dist.ndx"
    else:
        gmx_feed = "angle"
        process_input = "Angles"
        output_flag = "-ov"
        ndx_file_name = "angle.ndx"

    ensure_dir(out_dir)

    def _run_one(traj_file, ndx_dir_for_rep, rep_idx):
        ndx_file = os.path.join(ndx_dir_for_rep, ndx_file_name)
        if not os.path.exists(ndx_file):
            raise FileNotFoundError(f"Missing NDX: {ndx_file}")
        tag = f"_{rep_idx}" if rep_idx is not None else (f"_{cutoff}" if cutoff is not None else "")
        out_file = os.path.join(out_dir, f"{pdb}_{ligand}_{measure}{tag}.xvg")
        cmd = [gmx_command, gmx_feed, "-f", traj_file, "-n", ndx_file, output_flag, out_file, "-nobackup"]
        run_subprocess(cmd, stdin=f"{process_input}\n", check=True)
        return out_file
    
    if isinstance(traj_override, str):
        if ndx_dir is None:
            raise ValueError("ndx_dir must be provided when traj_override (single) is used.")
        out_file = _run_one(traj_override, ndx_dir, rep_idx=None)
        return out_file

    per_repeat_arrays= []
    per_repeat_outfiles = []

    for repeat in range(number_repeats):
        if isinstance(traj_override, list):
            traj_file = traj_override[repeat]
            ndx_dir_for_rep = ndx_dir if ndx_dir is not None else os.path.dirname(traj_file)
        else:
            ndx_dir_for_rep = os.path.join(master_dir, job_name, ligand, pdb, f"REP{repeat}")
            traj_file       = os.path.join(ndx_dir_for_rep, traj_name)

        out_file = _run_one(traj_file, ndx_dir_for_rep, rep_idx=repeat)
        per_repeat_outfiles.append(out_file)

        arr = np.loadtxt(out_file, comments=["@","#"])
        per_repeat_arrays.append(arr)

    
    if combine_repeats:
        data = np.vstack(per_repeat_arrays)
        combined_file = os.path.join(out_dir, f"{pdb}_{ligand}_{measure}.xvg")
        np.savetxt(combined_file, data)
        return combined_file
    
    else:
        return per_repeat_outfiles

def read_measure(xvg_file, measure, pdb, ligand,
        repeat_labels: list[int] | None = None):
    from io import StringIO

    if measure not in ("dist", "angle"):
        raise ValueError("measure must be 'dist' or 'angle'")
    print(f"reading measure {measure}")

    ligand_name = 'Unliganded' if ligand == 'apo' else ligand.upper()
    col_names = ["time", "distance"] if measure == 'dist' else ["time", "angle"]

    def _load_single_file(fname: str) -> pd.DataFrame:
        try:
            data = np.loadtxt(fname, comments=('#','@'))
        except TypeError:
            with open(xvg_file) as f:
                lines = [ln for ln in f if not ln.lstrip().startswith(('#', '@'))]
            data = np.loadtxt(StringIO(''.join(lines)))
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        df = pd.DataFrame(data, columns=col_names)
    #df = pd.read_table(xvg_file, delim_whitespace=True,
    #        header=None, names = col_names, 
    #        skiprows=lambda x: open(xvg_file).read().startswith(('#','@')))

        df.insert(2, "system", [pdb.upper()]*len(df))
        df.insert(2, "ligand", [ligand_name.upper()]*len(df))
        #print(df) 
        return df[["distance", "system", "ligand"]] if measure == 'dist' else df[["angle", "system", "ligand"]]

    if isinstance(xvg_file, str):
        df = _load_single_file(xvg_file)
    elif isinstance(xvg_file, list):
        dfs = []
        for i, fname in enumerate(xvg_file):
            dfi = _load_single_file(fname)

            dfi["repeat"] = (
                    repeat_labels[i] if repeat_labels and i < len(repeat_labels) else i
                    )
            dfs.append(dfi)
        df = pd.concat(dfs, ignore_index=True)

    cols = ['system', 'ligand']
    if "repeat" in df.columns:
        cols.append("repeat")

    return df[["distance", *cols]] if measure == 'dist' else df [["angle", *cols]]

def measure_plot(
        df,
        xcol,
        measure,
        pdb,
        ligand,
        out_png,
        title,
        xlim = None,
        ylim = None):
    if measure == 'dist':
        xlim = [0, 25]   if xlim is None else xlim
        ylim = [0, 0.15] if ylim is None else ylim
    elif measure == 'angle':
        xlim = [0, 180]  if xlim is None else xlim
        ylim = [0, 0.025] if ylim is None else ylim

    df["identifier"] = df["system"] + " " + df["ligand"]

    filtered_df = df[
            (df["system"] == pdb.upper())
            ]
    sns.kdeplot(data=filtered_df, x=xcol)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def dist_and_angle_run(
    pdb,
    ligand,
    master_dir,
    cutoff = None,
    number_repeats = NUMBER_REPEATS,
    traj_name = TRAJ_NAME,
    gmx_command = GMX_COMMAND,
    job_name = JOB_NAME,
    traj_override = None,
    out_png= None,
    analysis_dir: Optional[str] = None,
    do_plots = True,
    combine_repeats: bool = True,
    prev_gro_dir = PREV_GRO_DIR
    ):
    md, default_analysis = proj_paths(master_dir)
    analysis = analysis_dir or default_analysis 
    #make ndx files 
    res2 = len(pdb)
    selection_dist = f"ri1&aCA|ri{res2}&aCA"
    selection_angle_seq = [
        f"ri1&aCA|ri{res2}&aCA",
        f"ri5&aN",
        ]
    selection_signature= f"r_1_&_CA_r_{res2}_&_CA"

    dist_outdir = os.path.join(md, "end2end_dist")
    dist_outfile = os.path.join(dist_outdir, f"{pdb}_{ligand}_dist.xvg")

    angle_outdir = os.path.join(md, "end2end_angle")
    angle_outfile = os.path.join(angle_outdir, f"{pdb}_{ligand}_angle.xvg")
    
    if traj_override is None:

        make_ndx_for_pair(pdb, ligand, gmx_command, number_repeats, selection_dist, selection_angle_seq, selection_signature, job_name)

        dist_file = run_gmx_measure(
            pdb, ligand, gmx_command, number_repeats, traj_name, job_name, "dist", dist_outdir, master_dir, combine_repeats = combine_repeats
            )

        angle_file = run_gmx_measure(
            pdb, ligand, gmx_command, number_repeats, traj_name, job_name, "angle", angle_outdir, master_dir, combine_repeats=combine_repeats
            )

    else:
        pdb_dir = os.path.join(md, job_name, ligand, pdb)
        gro_file = os.path.join(pdb_dir, "REP0", f"{job_name}.gro")
        if not os.path.exists(gro_file):
            try:
                gro_file = glob.glob(os.path.join(pdb_dir, "REP0", "*.gro"))[0]
            except IndexError as e:
                print(f"{e}: gro_file not found in {job_name} run, falling back to previous directory")
                gro_file = glob.glob(os.path.join(md, prev_gro_dir, ligand, pdb, "REP0", "*.gro"))[0]
        print(gro_file)
        dist_ndx, angle_ndx = make_ndx_for_pair_in_dir(
                gro_file=gro_file,
                out_dir=pdb_dir,
                gmx_command=gmx_command,
                selection_dist=selection_dist,
                selection_angle_seq = selection_angle_seq,
                selection_string_signature=selection_signature)
        dist_file = run_gmx_measure(
                pdb, ligand, gmx_command, number_repeats, traj_name, job_name,
                "dist", dist_outdir, md, cutoff,
                traj_override=traj_override, ndx_dir=pdb_dir, combine_repeats=combine_repeats)
        angle_file = run_gmx_measure(
                pdb, ligand, gmx_command, number_repeats, traj_name, job_name,
                "angle", angle_outdir, md, cutoff,
                traj_override=traj_override, ndx_dir=pdb_dir, combine_repeats=combine_repeats)


    ligand_name = "Unliganded" if ligand == "apo" else ligand.upper()
    df_dist = read_measure(dist_file, measure='dist', pdb=pdb, ligand=ligand)
    df_angle = read_measure(angle_file, measure='angle', pdb=pdb, ligand=ligand)

    # read the measures table
    df_dist = df_dist.assign(distance=lambda d: d["distance"] * 10.0)
    df_dist["identifier"] = df_dist["system"] + " " + df_dist["ligand"]
    df_angle["identifier"] = df_angle["system"] + " " + df_angle["ligand"]
    
    if out_png is None:
        out_png_dist=os.path.join(analysis, f"{pdb}_{ligand}_dist.png")
        out_png_angle=os.path.join(analysis, f"{pdb}_{ligand}_angle.png")
    else:
        dist_png = f"dist_{out_png}"
        out_png_dist = os.path.join(analysis, dist_png)
        angle_png = f"angle_{out_png}"
        out_png_angle = os.path.join(analysis, angle_png)
    # make plots
    if do_plots:
        if combine_repeats or "repeat" not in df_dist.columns:
            measure_plot(
                    df = df_dist[df_dist["system"] == pdb.upper()],
                    xcol = "distance",
                    measure = "dist",
                    pdb = pdb,
                    ligand = ligand,
                    out_png=out_png_dist,
                    title=f"{pdb.upper()} {ligand_name} - Distance"
                    )
            measure_plot(
                    df = df_angle[df_angle["system"] == pdb.upper()],
                    xcol = "angle",
                    measure = "angle",
                    pdb = pdb,
                    ligand = ligand,
                    out_png=out_png_angle,
                    title=f"{pdb.upper()} {ligand_name} - Angle")
        else:
            for r in range(number_repeats):
                measure_plot(
                        df = df_dist[(df_dist["system"] == pdb.upper()) & (df_dist["repeat"] ==r)],
                        xcol="distance", measure="dist",
                        pdb=pdb, ligand=ligand,
                        out_png=os.path.join(analysis, f"{pdb}_{ligand}_dist_rep{r}.png"),
                        title=f"{pdb.upper()} {ligand_name} - Distance (REP{r})")
                measure_plot(
                    df = df_angle[(df_angle["system"] == pdb.upper()) & (df_angle["repeat"] == r)],
                    xcol = "angle",
                    measure = "angle",
                    pdb = pdb,
                    ligand = ligand,
                    out_png=os.path.join(analysis, f"{pdb}_{ligand}_angle_rep{r}.png"),
                    title=f"{pdb.upper()} {ligand_name} - Angle (REP{r})")

    return df_dist, df_angle

####################
#####NESTINESS######
####################

#---------Dihedral 
def safe_res_selections(residues, sel_func = "phi"):
    """Build list of angle-defining AtomGroups, filterint out None and non-4 length groups
    sel_func: 'phi' | 'psi'
    """

    ags = []
    for res in residues:
        ag = res.phi_selection() if sel_func == "phi" else res.psi_selection()
        if ag is not None and len(ag) == 4:
            ags.append(ag)
    return ags

def get_dihedrals(
    gro=None,
    xtc=None,
    res1=4,
    res2=7,
    u=None):
    """
    get phi/psi dihedrals for residues (python slide: inclusive start, exclusive end)
    """
    u = coerce_universe(gro=gro, trajectories=xtc, u=u)
    ags_phi = safe_res_selections(u.residues[res1:res2], sel_func="phi")
    ags_psi = safe_res_selections(u.residues[res1:res2], sel_func="psi")

    R_phi = Dihedral(ags_phi).run()
    R_psi = Dihedral(ags_psi).run()
    return R_phi, R_psi

def assign_LR(R_phi, R_psi, index):
    assignments = []
    for phi, psi in zip(R_phi.results.angles[index], R_psi.results.angles[index]):
        if (-140 < phi < -20) and (-90 < psi < 40):
            assignments.append("R")
        elif (20 < phi < 140) and (-40 < psi < 90):
            assignments.append("L")
        else:
            assignments.append("N")
    return assignments

def assignment_windows(assignment):
    patterns = {
        "rl": ["R","L"],
        "lr": ["L","R"],
        "rlr": ["R","L","R"],
        "lrl": ["L","R","L"],
        "lrlr": ["L","R","L","R"]
            }

    def eq_window(lst, pattern):
        return int(lst == pattern)

    rl   = np.zeros((1,3))
    lr   = np.zeros((1,3))
    rlr  = np.zeros((1,2))
    lrl  = np.zeros((1,2))
    lrlr = np.zeros((1,1))

    #2-char windows
    rl[0,0]   = eq_window(assignment[0:2], patterns["rl"])
    rl[0,1]   = eq_window(assignment[1:3], patterns["rl"])
    rl[0,2]   = eq_window(assignment[2:4], patterns["rl"])

    lr[0,0]   = eq_window(assignment[0:2], patterns["lr"])
    lr[0,1]   = eq_window(assignment[1:3], patterns["lr"])
    lr[0,2]   = eq_window(assignment[2:4], patterns["lr"])

    #3-char windows
    rlr[0,0]  = eq_window(assignment[0:3], patterns["rlr"])
    rlr[0,1]  = eq_window(assignment[1:4], patterns["rlr"])

    lrl[0,0]  = eq_window(assignment[0:3], patterns["lrl"])
    lrl[0,1]  = eq_window(assignment[1:4], patterns["lrl"])

    #4-char windows
    lrlr[0,0] = eq_window(assignment[0:4], patterns["lrlr"])

    return rl, lr, rlr, lrl, lrlr

def summarize_assignment_fractions(assignments):
    """
    Given a list of indicator arrays for one pattern
    stack, sum, and divide by N to get fractions
    """
    arr = np.array(assignments)
    summed = arr.sum(axis=0)
    frac = summed / len(assignments)
    return frac.reshape(1, -1)

def plot_nestiness_grid(
        fractions,
        titles,
        fig_title,
        out_png
        ):
    fig, axs = plt.subplots(1, len(fractions), figsize=(5,2))
    bounds=[0,0.1,1,10,50,100]
    norm = mcolors.BoundaryNorm(bounds, ncolors=256)
    cmap = plt.cm.Blues
    im = None

    for j, fr in enumerate(fractions):
        ax = axs[j]
        #im = ax.imshow(fr, cmap=cmap, vmin=0, vmax=1, aspect='auto')
        im = ax.imshow(fr, cmap=cmap, norm=norm, aspect='auto')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(0.38)
        ax.set_title(titles[j])

    fig.suptitle(fig_title)
    fig.subplots_adjust(right=0.9, wspace=0.1, hspace=0.05)
    
    cbar_ax = fig.add_axes([0.93, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax,boundaries=bounds,ticks=bounds)
    #cbar.set_ticks([0, 0.5, 1])
    #cbar.set_ticks([0, 0.1, 1, 10, 50, 100])
    cbar.set_label("Fraction of \nSimulation Time")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()

def dihedral_nestiness_figure(
        pdb,
        ligand,
        master_dir,
        res1 = 4,
        res2 = 7,
        prev_gro_name=PREV_GRO_NAME,
        job_name=JOB_NAME,
        number_repeats=NUMBER_REPEATS,
        out_png = None,
        traj_name = TRAJ_NAME,
        use_cutoff_xtc=False,
        cutoff=None,
        xtc_override=None,
        u=None):
    md, analysis_dir = proj_paths(master_dir)
    ensure_dir(analysis_dir)
    out_png = os.path.join(analysis_dir, out_png) if out_png is not None else os.path.join(analysis_dir, f"nest_classify_{pdb}_{ligand}.png")

    pdb_dir = os.path.join(md, job_name, ligand, pdb)
    gro = os.path.join(md, job_name, ligand, pdb, f"REP0", f"{prev_gro_name}.gro")
    #trajectories = [
    #        os.path.join(md, job_name, ligand, pdb, f"REP{r}", traj_name)
    #        for r in range(number_repeats)
    #        ]
    if u is not None:
        R_phi, R_psi = get_dihedrals(res1=res1, res2=res2, u=u)
    else:
        if xtc_override is not None:
            trajectories = [xtc_override]
        elif use_cutoff_xtc:
            if cutoff is None:
                raise ValueError("cutoff must be provided when use_cutoff_xtc is not False")
            cutoff_xtc = os.path.join(pdb_dir, f"adsorbed_c{cutoff}.xtc")
            if not os.path.exists(cutoff_xtc):
                raise FileNotFoundError(f"{cutoff_xtc} does not exist")
            trajectories = [cutoff_xtc]

        else:
            trajectories = [
                    os.path.join(pdb_dir, f"REP{r}", traj_name)
                    for r in range(number_repeats)
                    ]
        R_phi, R_psi = get_dihedrals(gro, trajectories, res1=res1, res2=res2)


    #make the assignments array
    assignments_rl, assignments_lr, assignments_rlr, assignments_lrl, assignments_lrlr = [], [], [], [], []
    n_frames = len(R_phi.results.angles)
    for i in range(n_frames):
        assignment = assign_LR(R_phi, R_psi, i)
        rl, lr, rlr, lrl, lrlr = assignment_windows(assignment)
        assignments_rl.append(rl)
        assignments_lr.append(lr)
        assignments_rlr.append(rlr)
        assignments_lrl.append(lrl)
        assignments_lrlr.append(lrlr)

    fractions = [
            summarize_assignment_fractions(assignments_rl),
            summarize_assignment_fractions(assignments_lr),
            summarize_assignment_fractions(assignments_rlr),
            summarize_assignment_fractions(assignments_lrl),
            summarize_assignment_fractions(assignments_lrlr)
            ]
    titles = ["RL", "LR", "RLR", "LRL", "LRLR"]
    ligand_name = "Unliganded" if ligand == "apo" else ligand.upper()
    if cutoff is None:
        fig_title = f"{pdb.upper()} {ligand_name}"
    else:
        fig_title = f"{pdb.upper()} {ligand_name} {cutoff}$\\AA$"
    
    plot_nestiness_grid(fractions, titles, fig_title, out_png)

############################################
############RADIUS OF GYRATION##############
############################################

def radgyr(atomgroup, masses, total_mass=None):
    coordinates = atomgroup.positions
    com    = atomgroup.center_of_mass()
    ri_sq  = (coordinates - com) ** 2
    sq     = np.sum(ri_sq, axis =1)
    sq_x   = np.sum(ri_sq[:, [1, 2]], axis=1)
    sq_y   = np.sum(ri_sq[:, [0, 2]], axis=1)
    sq_z   = np.sum(ri_sq[:, [0, 1]], axis=1)
    sq_rs  = np.array([sq, sq_x, sq_y, sq_z])
    rog_sq = np.sum(masses * sq_rs, axis=1) / total_mass
    # return 4 rofg values first is 3D, others are projection
    return np.sqrt(rog_sq)

def rog_timeseries(u, selection="protein"): #SETUP TO READ UNIVERSE CONSTRUCTED PREVIOUSLY IN DIHEDRAL WORK
    from MDAnalysis.analysis.base import AnalysisFromFunction

    ag  = u.select_atoms(selection)
    ana = AnalysisFromFunction(
        radgyr, u.trajectory, ag, ag.masses, total_mass=np.sum(ag.masses)
    ).run()
    times_ps = ana.results.get("times", np.arange(ana.results['timeseries'].shape[0]))
    data_nm  = ana.results['timeseries']
    return np.asarray(times_ps), np.asarray(data_nm)

def resample_to_grid(times_list, data_list, npts=1000):
    t_end = min(t[-1] for t in times_list)
    grid  = np.linspace(0.0, t_end, npts)
    resampled = []
    for t, y in zip(times_list, data_list):
        y_interp = np.vstack([np.interp(grid, t, y[:, j]) for j in range(4)]).T
        resampled.append(y_interp)
    return grid, np.stack(resampled, axis=0)

def mean_and_ci_rgyr(y_repeats):
    mu = y_repeats.mean(axis=0)
    sd = y_repeats.std(axis=0, ddof=1)
    R  = y_repeats.shape[0]
    half = 1.96 * sd / np.sqrt(R)
    return mu, mu - half, mu + half

def plot_rog_multi(
        times_list, 
        data_list, 
        title, 
        savepath, 
        show_axes=("all",), 
        to_angstrom=True):
    grid, Y = resample_to_grid(times_list, data_list, npts=1000)
    grid_us = grid / 1e6

    factor = 10.0 if to_angstrom else 1.0
    Y *= factor
    ylab = r"Radius of gyration ($\AA$)" if to_angstrom else r"Radius of gyration (nm)"
    comp_idx = {"all": 0, "x": 1, "y": 2, "z": 3}
    comps = [c for c in ("all", "x", "y", "z") if c in show_axes]

    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=300)
    for c in comps:
        j = comp_idx[c]
        for r in range(Y.shape[0]):
            ax.plot(grid_us, Y[r, :, j], lw=0.8, alpha=0.3, color='gray', label="Replicates" if r == 0 else None)
        mu, lo, hi = mean_and_ci_rgyr(Y[:, :, j])
        ax.plot(grid_us, mu, lw=2.0, label="Mean", color='blue')
        ax.fill_between(grid_us, lo, hi, alpha=0.2, color='blue', label="CI")

    ax.set_xlabel("Time (µs)")
    ax.set_ylabel(ylab)
    if title:
        ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=False)
    fig.tight_layout()
    fig.savefig(savepath, dpi=300)
    print(f"[SAVED] RoG plot -> {savepath}")
    plt.close(fig)

def rog_run(
        pdb,
        ligand,
        master_dir,
        cutoff = None,
        number_repeats = NUMBER_REPEATS,
        prev_gro_name = PREV_GRO_NAME,
        traj_name = TRAJ_NAME,
        out_png = None,
        use_cutoff_xtc=False,
        job_name=None,
        #xtc_override=None,
        u=None,
        surface=False):
    
    times_list, data_list = [], []

    if ligand == 'apo':
        ligand_name = "unliganded"
    else:
        ligand_name = ligand

    md, analysis_dir = proj_paths(master_dir)
    ensure_dir(analysis_dir)

    pdb_dir = os.path.join(md, job_name, ligand, pdb)
    
    gro = os.path.join(md, job_name, ligand, pdb, f"REP0", f"{prev_gro_name}.gro")

    if not os.path.isfile(gro):
        print("Using npt gro")
        npt_dir = os.path.join(master_directory, 'npt', ligand, pdb, 'REP0')
        gro = glob.glob(os.path.join(npt_dir, "*.gro"))[0]

    if u is not None:
        t, y = rog_timeseries(u)
    else:
        #if xtc_override is not None:
        #    trajectories = [xtc_override]
        #elif use_cutoff_xtc:
        if cutoff is None:
            raise ValueError("cutoff must be provided when use_cutoff_xtc is not False")
        for repeat in range(number_repeats):
            rep_dir = os.path.join(pdb_dir, f"REP{repeat}")
            if surface is not False:
                traj_file = os.path.join(rep_dir, f"adsorbed_c{cutoff}.xtc")
                out_png = os.path.join(analysis_dir, f"ROG_{pdb}_{ligand}_c{cutoff}.png")
            else:
                traj_file = os.path.join(rep_dir, traj_name)
                out_png = os.path.join(analysis_dir, out_png) if out_png is not None else os.path.join(analysis_dir, f"ROG_classify_{pdb}_{ligand}.png")
            u = mda.Universe(gro, traj_file)
            t, y = rog_timeseries(u)
            times_list.append(t)
            data_list.append(y)

        if times_list:
            plot_rog_multi(times_list, data_list,
                    title=f"RoG: {pdb.upper()} {ligand_name.upper()}",
                    savepath = out_png, #FIX THIS
                    show_axes=("all",),
                    to_angstrom=True)
            rog_results[(pdb, ligand)] = (times_list, data_list)

    return rog_results



############################
#####      ABEGO     #######
############################

# TODO: CURRENTLY WORKS ONLY FOR A SINGLE REPEAT. WANT IT TO WORK ON COLLECTION OF REPEATS


def classify_one_abego(phi, psi):
    if phi >= 0:
        if (-100 <= psi and psi < 100):
            return 4 # G (alpha-L)
        else:
            return 3 # E
    else:
        if (-75 <= psi and psi < 50):
            return 1 # A (helical)
        else:
            return 2 # B (beta)

def classify_all_abego(phi_angles:np.ndarray, psi_angles:np.ndarray) -> np.ndarray:
    abego = np.zeros((phi_angles.shape))

    for frame in range(phi_angles.shape[0]):
        for i, (phi, psi) in enumerate(zip(phi_angles[frame,:], psi_angles[frame,:])):
            abego[frame,i] = classify_one_abego(phi,psi)

    return abego

def plot_abego(pdb, ligand, abego_list, analysis_directory = None, labels=None, cutoff=None, out_png=None, num_residues = 8):
    if num_residues == 8:
        template = [2,2,4,1,4,1]
        titles = ['B', 'B', 'G', 'A', 'G', 'A']
    else:
        template = [3,2,2,4,1,4,1,1]
        titles = ['E', 'B', 'B', 'G', 'A', 'G', 'A', 'A']

    nrows = len(abego_list)
    ncols = len(template)

    fig, axs = plt.subplots(nrows, ncols, figsize = (5,2), tight_layout=False)

    fig.subplots_adjust(wspace=0, hspace=0)
    cmap = plt.cm.gray_r

    if nrows == 1:
        axs = np.array([axs])
    for abego_ind, abego in enumerate(abego_list):
        for i, val in enumerate(template):
            count = np.sum(abego[:,i] == val) / len(abego)

            im = axs[abego_ind, i].imshow([[count]], cmap=cmap, vmin=0, vmax=1.0, aspect='equal')
            axs[abego_ind, i].set_xticks([])
            axs[abego_ind, i].set_yticks([])

    for i, t in enumerate(titles):
        axs[0, i].set_title(t)

    if labels is None:
        labels = [f"{pdb}_{ligand}"]

    for i, label in enumerate(labels):
        axs[i,0].set_ylabel(label)

    cbar_ax = fig.add_axes([1.0, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Simulation Time")
    if cutoff is None:
        output = os.path.join(analysis_directory, f"abego_{pdb}_{ligand}.png")
    else:
        output = os.path.join(analysis_directory, f"abego_{pdb}_{ligand}_c{cutoff}.png")
    fig.savefig(output, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"[SAVED] abego figure in {output}")

def process_abego(master_dir,
                pdb,
                ligand,
                job_name = JOB_NAME,
                prev_gro_name = PREV_GRO_NAME,
                number_repeats = NUMBER_REPEATS,
                traj_name = TRAJ_NAME,
                indexing_pairs = None,
                cutoff = None,
                labels=None,
                traj_override=None,
                ): 
    abego_list = []

    analysis_directory = os.path.join(master_dir, 'analysis')
    pdb_dir = os.path.join(master_dir, job_name, ligand, pdb)

    if cutoff is None:
        print(f"[DEBUG] No cutoff supplied")
        for repeat in number_repeats:
            gro = os.path.join(pdb_dir, f'REP{repeat}', prev_gro_name)
            if not os.path.exists(gro):
                try:
                    gro = glob.glob(os.path.join(pdb_dir, f'REP{repeat}', "*.gro"))[0]
                except IndexError as e:
                    print(f"{e}: gro_file not found in {job_name} run, falling back to npt directory")

                    gro = glob.glob(os.path.join(master_dir, 'npt', ligand, pdb, "REP0", "*.gro"))[0]
        
            xtc = os.path.join(pdb_dir, f'REP{repeat}', traj_name)

            for start, stop in indexing_pairs:
                phi_angles, psi_angles = get_dihedrals(gro, xtc, stop, start)
                abego_list.append(classify_all_abego(phi_angles, psi_angles))

    else:
        gro = os.path.join(pdb_dir, f'REP0', prev_gro_name)
        if not os.path.isfile(gro):
            #print(f"{e}: gro_file not found in {job_name} run, falling back to npt directory")
            npt_dir = os.path.join(master_dir, 'npt', ligand, pdb, "REP0")
            gro = glob.glob(os.path.join(npt_dir, "*.gro"))[0]

        if traj_override is None:
            xtc = os.path.join(pdb_dir, f"adsorbed_c{cutoff}.xtc")
        else:
            xtc = traj_override if os.path.isabs(traj_override) else os.path.join(pdb_dir, traj_override)
        for start, stop in indexing_pairs:
            phi_angles, psi_angles = get_dihedrals(gro, xtc, res1=start, res2=stop)
            phi_angles = phi_angles.results.angles
            psi_angles = psi_angles.results.angles
            abego_list.append(classify_all_abego(phi_angles, psi_angles))

        print(f"[DEBUG] {abego_list}")


    plot_abego(pdb, ligand, abego_list, analysis_directory = analysis_directory, labels=labels, cutoff=cutoff)


#########################################
############     RMSD     ###############
#########################################

def setup_rmsd_figure():
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, sharey=True)
    ax2.set_xlabel(r"time (ps)")
    ax2.set_ylabel(r"Sidechain RMSD ($\AA$)")
    ax1.set_ylabel(r"Backbone RMSD ($\AA$)")

    return fig, ax1, ax2

def compute_rmsd(
    universe: mda.Universe,
    reference_universe: mda.Universe,
    selection: str,
):
    """
    Compute RMSD for a given atom selection.

    Returns
    -------
    time : np.ndarray
        Time points from MDAnalysis RMSD result.
    rmsd_values : np.ndarray
        RMSD values in Å.
    """
    import MDAnalysis.analysis.rms as rms

    R = rms.RMSD(universe, reference_universe, select=selection)
    R.run()
    data = R.rmsd.T  # shape (4, nframes): [frame, time(ps), rmsd, rmsd_ref]
    time = data[1]
    rmsd_values = data[2]
    return time, rmsd_values

def plot_single_replica_rmsd(
    ax_bb,
    ax_sc,
    time_bb,
    rmsd_bb,
    time_sc,
    rmsd_sc,
    repeat: int,
):
    """Plot backbone and sidechain RMSD for a single replica."""
    ax_bb.plot(time_bb, rmsd_bb, label=f'REP{repeat}')
    ax_sc.plot(time_sc, rmsd_sc)


def finalize_rmsd_figure(
    fig,
    ax_bb,
    ligand_name: str,
    pdb: str,
    master_directory: str,
    out_folder: str = 'analysis',
    cutoff = None,
):
    """
    Set title, legend, limits, and save the figure.
    """
    ligand_string = get_ligand_string(ligand_name)
    fig.suptitle(f'RMSD for {pdb} {ligand_string}')
    ax_bb.legend(loc='best')
    plt.tight_layout()
    ax_bb.set_ylim(0, 6.5)

    out_dir = os.path.join(master_directory, out_folder)
    os.makedirs(out_dir, exist_ok=True)
    if cutoff is not None:
        out_png = os.path.join(out_dir, f'{pdb}_{ligand_name}_c{cutoff}_rmsd.png')
    else:
        out_png = os.path.join(out_dir, f'{pdb}_{ligand_name}_rmsd.png')
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def rmsd(
    ligand_name: str,
    pdb: str,
    master_directory: str,
    job_name: str,
    number_of_repeats: int,
    traj_override: str | None = None,
    cutoff = None
):
    """
    High-level RMSD driver function.

    Parameters
    ----------
    ligand_name : str
    pdb : str
    master_directory : str
    job_name : str
        Name of the production job ("prod", etc.). Used for default trajectory path.
    number_of_repeats : int
    traj_override : str or None
        Optional trajectory override for adsorption analysis or alternate workflows.
        See get_trajectory_path() for behavior.

        Examples:
        ---------
        1) Use a different trajectory name in the same directory:
           traj_override="adsorption.xtc"

        2) Use a different directory structure, one trajectory per replica:
           traj_override=os.path.join(
               master_directory,
               "adsorption",
               ligand_name,
               pdb,
               "REP{repeat}",
               "adsorption.xtc",
           )
           (Note: the {repeat} will be formatted automatically.)
    """

    print(f"[DEBUG] starting RMSD")
    # Reference
    rmsd_ref = get_reference_universe(master_directory, ligand_name, pdb)

    # Figure
    fig, ax_bb, ax_sc = setup_rmsd_figure()
    print("[DEBUG] entering into repeat loop")
    # Loop over replicas
    for repeat in range(number_of_repeats):
        try:
            print("[DEBUG] FINDING GRO FILE")
            gro_file = find_topology_file(
                master_directory=master_directory,
                job_name=job_name,
                ligand_name=ligand_name,
                pdb=pdb,
                repeat=repeat,
            )
            print(f"[DEBUG] GRO FILE FOUND {gro_file}")
        except FileNotFoundError as e:
            print(f"[DEBUG] GRO FILE NOT FOUND")
            print(e)
            continue
        per_rep_traj_override = traj_override

        if traj_override is not None:
            if f"REP{repeat}" not in traj_override:
                traj_name = os.path.basename(traj_override)

                per_rep_traj_override = os.path.join(
                        master_directory,
                        job_name,
                        ligand_name,
                        pdb,
                        f"REP{repeat}",
                        traj_name)

        print(f'trajectory being used is {per_rep_traj_override}')
        traj_file = get_trajectory_path(
            master_directory=master_directory,
            job_name=job_name,
            ligand_name=ligand_name,
            pdb=pdb,
            repeat=repeat,
            traj_override=per_rep_traj_override,
        )

        if not os.path.exists(traj_file):
            print(f"[WARN] Missing trajectory for REP{repeat}: {traj_file}")
            continue

        u = mda.Universe(gro_file, traj_file)
        # Backbone RMSD
        time_bb, rmsd_bb = compute_rmsd(
            universe=u,
            reference_universe=rmsd_ref,
            selection="backbone",
        )

        # Sidechain RMSD
        time_sc, rmsd_sc = compute_rmsd(
            universe=u,
            reference_universe=rmsd_ref,
            selection="protein and not backbone and not name H*",
        )

        # Plot
        plot_single_replica_rmsd(
            ax_bb=ax_bb,
            ax_sc=ax_sc,
            time_bb=time_bb,
            rmsd_bb=rmsd_bb,
            time_sc=time_sc,
            rmsd_sc=rmsd_sc,
            repeat=repeat,
        )
    # Finalize & save
    finalize_rmsd_figure(
        fig=fig,
        ax_bb=ax_bb,
        ligand_name=ligand_name,
        pdb=pdb,
        master_directory=master_directory,
        cutoff=cutoff
    )

########################################
############   CONTACTS   ##############
########################################

def system_ligand_to_resname(ligand: str) -> str | None:
    """
    Map system-level ligand label (e.g. 'apo', 'hpo4', 'gtp', 'atp')
    to the residue name used in topology and hb_groups (e.g. 'PO4', 'GTP', 'ATP').

    Returns
    -------
    str or None
        Residue name used in the structure / hb_lig dict.
        Returns None for apo / unliganded systems.
    """
    ligand = ligand.lower()
    if ligand == "apo":
        return None
    if ligand == "hpo4":
        return "PO4"
    if ligand == "gtp":
        return "GTP"
    if ligand == "atp":
        return "ATP"
    # you can extend this if needed
    raise ValueError(f"[system_ligand_to_resname] Unknown ligand label: {ligand}")

def calc_da_pairs_for_residue(
    pdb: str,
    residue_index: int,
    ligand_resname: str,
    group: str,
    hb_aa_b=aa_b,
    hb_aa_s=aa_s,
    hb_lig_defs=lig,
    hb_aa_abbr=aa_abbr,
) -> list[list[str]]:
    """
    Find all possible donor–acceptor pairs between residue `residue_index`
    in peptide sequence `pdb` and chemical group `group` on ligand `ligand_resname`.

    Parameters
    ----------
    pdb : str
        Peptide sequence string, e.g. 'gpsgsgks'.
    residue_index : int
        0-based index into `pdb`.
    ligand_resname : str
        Ligand residue name used in hb_groups / topology, e.g. 'PO4', 'GTP', 'ATP'.
    group : str
        Chemical group label on the ligand, e.g. 'Phos', 'a', 'b', 'g', 'base', 'ribose'.

    Returns
    -------
    list[list[str]]
        Each element is [donor_string, acceptor_atomname]
        where donor_string is 'ATOM HE' like in the original script.
    """
    pairs: list[list[str]] = []
    res_one = pdb[residue_index]
    res_three = hb_aa_abbr[res_one]

    # --- Side chain contributions ---
    # Peptide sidechain donor -> ligand acceptor
    for d in hb_aa_s[res_three]["D"]:
        for a in hb_lig_defs[ligand_resname][group]["A"]:
            pairs.append([d, a])

    # Peptide sidechain acceptor -> ligand donor
    for a in hb_aa_s[res_three]["A"]:
        for d in hb_lig_defs[ligand_resname][group]["D"]:
            pairs.append([d, a])

    # --- Backbone -NH- donors ---
    if residue_index == 0:
        # N-terminus
        if res_three == "PRO":
            backbone_donors = hb_aa_b["Imine"]["D"]
        else:
            backbone_donors = hb_aa_b["Amine"]["D"]
    else:
        if res_three == "PRO":
            backbone_donors = hb_aa_b["Imide"]["D"]
        else:
            backbone_donors = hb_aa_b["Amide"]["D"]

    for d in backbone_donors:
        for a in hb_lig_defs[ligand_resname][group]["A"]:
            pairs.append([d, a])

    # --- Backbone carbonyl / carboxyl acceptors ---
    if residue_index == len(pdb) - 1:
        backbone_acceptors = hb_aa_b["Carboxyl"]["A"]
    else:
        backbone_acceptors = hb_aa_b["Carbonyl"]["A"]

    for a in backbone_acceptors:
        for d in hb_lig_defs[ligand_resname][group]["D"]:
            pairs.append([d, a])

    return pairs

def build_hbond_index_for_ligand_group(
    master_dir: str,
    job_name: str,
    ligand: str,
    pdb: str,
    ligand_resname: str,
    group: str,
    prev_gro_name: str = PREV_GRO_NAME,
    prev_gro_dir: str = PREV_GRO_DIR,
    write_file: bool = True,
) -> tuple[str | None, list[str], list[str]]:
    """
    Build a GROMACS index file with [Distances] and [Angles] groups containing
    all possible donor–acceptor pairs between the peptide `pdb` and the
    ligand group `group` on `ligand_resname`.

    This is the nestiness-ified replacement for the old `addition_to_ndx`.

    Parameters
    ----------
    master_dir : str
        Project root (typically from proj_paths()).
    job_name : str
        Name of the production job, e.g. 'prod'.
    ligand : str
        System ligand label, e.g. 'apo', 'hpo4', 'gtp', 'atp'.
    pdb : str
        Peptide sequence label, e.g. 'gpsgsgks'.
    ligand_resname : str
        Ligand residue name used in topology / hb_groups, e.g. 'PO4', 'GTP', 'ATP'.
    group : str
        Chemical group on the ligand (e.g. 'Phos', 'a', 'b', 'g', 'base', 'ribose').
    prev_gro_name : str
        Basename of the reference .gro (e.g. 'prod' or 'npt'), without extension.
    prev_gro_dir : str
        Name of directory where previous step structures live, e.g. 'npt'.
    write_file : bool
        If True, writes the ndx to disk and returns its path.
        If False, just returns the headers and None for the path.

    Returns
    -------
    ndx_path : str or None
        Path to the created index file, or None if write_file=False.
    dist_header : list[str]
        Labels for distance columns (used later when reading XVG).
    angle_header : list[str]
        Labels for angle columns (used later when reading XVG).
    """
    import glob

    pdb_dir = os.path.join(master_dir, job_name, ligand, pdb, "REP0")
    # First, try prod/npt-style gro
    gro = os.path.join(pdb_dir, f"{prev_gro_name}.gro")
    if not os.path.exists(gro):
        try:
            gro = glob.glob(os.path.join(pdb_dir, "*.gro"))[0]
        except IndexError:
            # Fall back to previous directory (e.g. npt)
            prev_dir = os.path.join(master_dir, prev_gro_dir, ligand, pdb, "REP0")
            gro = glob.glob(os.path.join(prev_dir, "*.gro"))[0]

    # Parse the .gro file to build maps for peptide + ligand atoms
    pep_atoms: dict[int, tuple[str, dict[str, str]]] = {}
    lig_atoms: dict[str, str] = {}

    with open(gro, "r") as f:
        lines = f.readlines()

    # skip title, atom count, and the last box line
    for line in lines[2:-1]:
        if len(line) <= 40:
            continue
        try:
            resi = int(line[0:5].strip())
        except ValueError:
            continue
        resn = line[5:8].strip()
        atomn = line[11:15].strip()
        atomi = line[15:20].strip()

        if resn in hb_aa_s.keys():
            # peptide residue
            try:
                pep_atoms[resi][1][atomn] = atomi
            except KeyError:
                pep_atoms[resi] = (resn, {atomn: atomi})
        elif resn == ligand_resname:
            # ligand residue
            lig_atoms[atomn] = atomi

    if not pep_atoms:
        raise RuntimeError(
            f"[build_hbond_index_for_ligand_group] No peptide residues detected in {gro} "
            f"for pdb={pdb}"
        )
    if not lig_atoms:
        raise RuntimeError(
            f"[build_hbond_index_for_ligand_group] No ligand residues with resname={ligand_resname} "
            f"found in {gro}"
        )

    # We map position in sequence (0..len(pdb)-1) -> sorted residue IDs in structure
    sorted_resids = sorted(pep_atoms.keys())

    dist_entries: list[list[str]] = []
    angle_entries: list[list[str]] = []
    dist_header: list[str] = []
    angle_header: list[str] = []

    for i in range(len(pdb)):
        resi_gro = sorted_resids[i]
        resn_three, atom_dict = pep_atoms[resi_gro]
        da_pairs = calc_da_pairs_for_residue(
            pdb=pdb,
            residue_index=i,
            ligand_resname=ligand_resname,
            group=group,
        )

        for pair in da_pairs:
            donor_str, acceptor_name = pair
            donor_heavy_name = donor_str.split()[0]

            # angle triple: (heavy_donor, hydrogen, acceptor)
            angle = (donor_heavy_name, donor_str.split()[1], acceptor_name)

            # Decide which residue/ligand each atom lives in
            # donor heavy atom
            if angle[0] in atom_dict:
                a1 = atom_dict[angle[0]]
            else:
                a1 = lig_atoms.get(angle[0])

            # hydrogen
            if angle[1] in atom_dict:
                a2 = atom_dict[angle[1]]
            else:
                a2 = lig_atoms.get(angle[1])

            # acceptor
            if angle[2] in lig_atoms:
                a3 = lig_atoms[angle[2]]
            else:
                a3 = atom_dict.get(angle[2])

            if a1 is None or a2 is None or a3 is None:
                # Skip incomplete triplets
                continue

            angle_header.append(f"res {i+1} {angle[0]} {angle[1]} {angle[2]}")
            angle_entries.append([a1, a2, a3])

            # distance pair (heavy donor, acceptor)
            key = f"res {i+1} {donor_heavy_name} {acceptor_name}"
            if key in dist_header:
                continue

            if donor_heavy_name in atom_dict:
                d_heavy_idx = atom_dict[donor_heavy_name]
            else:
                d_heavy_idx = lig_atoms.get(donor_heavy_name)

            if acceptor_name in lig_atoms:
                acc_idx = lig_atoms[acceptor_name]
            else:
                acc_idx = atom_dict.get(acceptor_name)

            if d_heavy_idx is None or acc_idx is None:
                continue

            dist_header.append(key)
            dist_entries.append([d_heavy_idx, acc_idx])

    ndx_path: str | None = None
    if write_file:
        out_dir = os.path.join(master_dir, job_name, ligand, pdb)
        ensure_dir(out_dir)
        ndx_path = os.path.join(out_dir, f"dist_ang_{ligand_resname}_{group}.ndx")
        with open(ndx_path, "w") as f:
            f.write("[ Distances ]\n")
            for d in dist_entries:
                f.write(f"{d[0]} {d[1]}\n")
            f.write("[ Angles ]\n")
            for a in angle_entries:
                f.write(f"{a[0]} {a[1]} {a[2]}\n")

    return ndx_path, dist_header, angle_header

def run_gmx_contacts_for_group(
    pdb: str,
    ligand: str,
    ligand_resname: str,
    group: str,
    master_dir: str,
    job_name: str,
    number_repeats: int,
    traj_name: str,
    gmx_command: str = GMX_COMMAND,
    prev_gro_name: str = PREV_GRO_NAME,
    prev_gro_dir: str = PREV_GRO_DIR,
    traj_override: str | list[str] | None = None,
    combine_repeats: bool = True,
) -> tuple[str | list[str], str | list[str], list[str], list[str]]:
    """
    Run `gmx distance` and `gmx angle` for all donor/acceptor pairs between peptide `pdb`
    and ligand group `group` on `ligand_resname`.

    This is contacts-specific but follows the same override semantics as
    `run_gmx_measure` and the rest of the nestiness code.

    Parameters
    ----------
    pdb : str
    ligand : str
        System-level ligand label, e.g. 'apo', 'hpo4', 'gtp', 'atp'.
    ligand_resname : str
        Residue name for hb_groups/topology (e.g. 'PO4', 'GTP', 'ATP').
    group : str
        Ligand chemical group: 'Phos', 'a', 'b', 'g', 'base', 'ribose', etc.
    master_dir : str
    job_name : str
    number_repeats : int
    traj_name : str
        Default trajectory name when no override is given, e.g. 'prod_full.xtc'.
    gmx_command : str
    prev_gro_name : str
    prev_gro_dir : str
    traj_override : str | list[str] | None
        - None: use default per-REP trajs.
        - str: single override for all repeats (absolute or relative).
        - list[str]: one trajectory per repeat.
    combine_repeats : bool
        If True, concatenates arrays and writes one combined dist/angle xvg.
        Else returns per-REP xvg file lists.

    Returns
    -------
    dist_files : str or list[str]
        Path(s) to the distance .xvg output.
    angle_files : str or list[str]
        Path(s) to the angle .xvg output.
    dist_header : list[str]
        Column labels used for distance DataFrame.
    angle_header : list[str]
        Column labels used for angle DataFrame.
    """
    # 1) Build or reuse index file
    ndx_path, dist_header, angle_header = build_hbond_index_for_ligand_group(
        master_dir=master_dir,
        job_name=job_name,
        ligand=ligand,
        pdb=pdb,
        ligand_resname=ligand_resname,
        group=group,
        prev_gro_name=prev_gro_name,
        prev_gro_dir=prev_gro_dir,
        write_file=True,
    )

    if ndx_path is None:
        raise RuntimeError("[run_gmx_contacts_for_group] Failed to build ndx file")

    # 2) Prepare output directories
    dist_dir = os.path.join(master_dir, "contact_dist_xvg")
    angle_dir = os.path.join(master_dir, "contact_angle_xvg")
    ensure_dir(dist_dir)
    ensure_dir(angle_dir)

    # Helper to run for one replica / one traj
    def _run_one(rep_idx: int, traj_file: str) -> tuple[str, str]:
        dist_out = os.path.join(dist_dir, f"{pdb}_{ligand_resname}_{group}_dist_rep{rep_idx}.xvg")
        angle_out = os.path.join(angle_dir, f"{pdb}_{ligand_resname}_{group}_angle_rep{rep_idx}.xvg")

        # Distance
        cmd_dist = [
            gmx_command,
            "distance",
            "-f",
            traj_file,
            "-n",
            ndx_path,
            "-oall",
            dist_out,
        ]
        run_subprocess(cmd_dist, stdin="Distances\n", check=True)

        # Angle
        cmd_angle = [
            gmx_command,
            "angle",
            "-f",
            traj_file,
            "-n",
            ndx_path,
            "-all",
            "-ov",
            angle_out,
            "-type",
            "angle",
        ]
        run_subprocess(cmd_angle, stdin="Angles\n", check=True)

        return dist_out, angle_out

    dist_files_per_rep: list[str] = []
    angle_files_per_rep: list[str] = []

    for repeat in range(number_repeats):
        # Decide trajectory file for this REP
        if isinstance(traj_override, list):
            traj_file = traj_override[repeat]
        elif isinstance(traj_override, str):
            # If override is relative, join to the REP directory
            if os.path.isabs(traj_override):
                traj_file = traj_override
            else:
                rep_dir = os.path.join(master_dir, job_name, ligand, pdb, f"REP{repeat}")
                traj_file = os.path.join(rep_dir, os.path.basename(traj_override))
        else:
            rep_dir = os.path.join(master_dir, job_name, ligand, pdb, f"REP{repeat}")
            traj_file = os.path.join(rep_dir, traj_name)

        if not os.path.exists(traj_file):
            print(f"[WARN] Missing trajectory for contacts REP{repeat}: {traj_file}")
            continue

        dfile, afile = _run_one(repeat, traj_file)
        dist_files_per_rep.append(dfile)
        angle_files_per_rep.append(afile)

    if not combine_repeats:
        return dist_files_per_rep, angle_files_per_rep, dist_header, angle_header

    # Combine data across repeats
    dist_combined = None
    angle_combined = None

    for df in dist_files_per_rep:
        arr = np.loadtxt(df, comments=("#", "@"))
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        dist_combined = arr if dist_combined is None else np.vstack([dist_combined, arr])

    for af in angle_files_per_rep:
        arr = np.loadtxt(af, comments=("#", "@"))
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        angle_combined = arr if angle_combined is None else np.vstack([angle_combined, arr])

    dist_combined_file = os.path.join(dist_dir, f"{pdb}_{ligand_resname}_{group}_dist.xvg")
    angle_combined_file = os.path.join(angle_dir, f"{pdb}_{ligand_resname}_{group}_angle.xvg")

    if dist_combined is not None:
        np.savetxt(dist_combined_file, dist_combined)
    if angle_combined is not None:
        np.savetxt(angle_combined_file, angle_combined)

    return dist_combined_file, angle_combined_file, dist_header, angle_header

def _load_xvg_numeric(xvg_file: str) -> np.ndarray:
    """
    Load an XVG file as a numeric array, stripping GROMACS comment/header lines.
    Works for both raw GROMACS output and our combined text files.
    """
    from io import StringIO

    try:
        data = np.loadtxt(xvg_file, comments=("#", "@"))
    except Exception:
        # Fallback: manual strip of comment lines
        with open(xvg_file) as f:
            lines = [ln for ln in f if not ln.lstrip().startswith(("#", "@"))]
        data = np.loadtxt(StringIO("".join(lines)))

    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def read_contact_distances_xvg(
    xvg_files: str | list[str],
    dist_header: list[str],
) -> pd.DataFrame:
    """
    Read one or more distance XVG files produced by run_gmx_contacts_for_group
    into a single DataFrame.

    Columns: ['time'] + dist_header
    Multiple repeats are concatenated along the row dimension.
    """
    if isinstance(xvg_files, str):
        xvg_files = [xvg_files]

    dfs: list[pd.DataFrame] = []
    n_data_cols = 1 + len(dist_header)  # time + each distance column

    for fname in xvg_files:
        arr = _load_xvg_numeric(fname)
        if arr.shape[1] < n_data_cols:
            # If GROMACS added extra stuff, take first n_data_cols;
            # if combined file has exactly n_data_cols, this is safe.
            arr = arr[:, :n_data_cols]
        elif arr.shape[1] > n_data_cols:
            # More columns than expected: keep the first 1+len(header) and warn
            print(
                f"[WARN] Distance XVG {fname} has {arr.shape[1]} columns; "
                f"expected {n_data_cols}. Truncating extra columns."
            )
            arr = arr[:, :n_data_cols]

        col_names = ["time"] + dist_header
        df = pd.DataFrame(arr, columns=col_names)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=["time"] + dist_header)

    return pd.concat(dfs, ignore_index=True)


def read_contact_angles_xvg(
    xvg_files: str | list[str],
    angle_header: list[str],
) -> pd.DataFrame:
    """
    Read one or more angle XVG files produced by run_gmx_contacts_for_group
    into a single DataFrame.

    Assumes layout: time, filler, angle_1, angle_2, ...
    (matching the original contacts.py behavior with `-all -ov`).

    Columns: ['time', 'filler'] + angle_header
    """
    if isinstance(xvg_files, str):
        xvg_files = [xvg_files]

    dfs: list[pd.DataFrame] = []
    n_data_cols = 2 + len(angle_header)  # time + filler + each angle column

    for fname in xvg_files:
        arr = _load_xvg_numeric(fname)
        if arr.shape[1] < n_data_cols:
            arr = arr[:, :n_data_cols]
        elif arr.shape[1] > n_data_cols:
            print(
                f"[WARN] Angle XVG {fname} has {arr.shape[1]} columns; "
                f"expected {n_data_cols}. Truncating extra columns."
            )
            arr = arr[:, :n_data_cols]

        col_names = ["time", "filler"] + angle_header
        df = pd.DataFrame(arr, columns=col_names)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=["time", "filler"] + angle_header)

    return pd.concat(dfs, ignore_index=True)

def calc_contacts_for_group(
    pdb: str,
    ligand: str,
    ligand_resname: str,
    group: str,
    master_dir: str,
    job_name: str,
    number_repeats: int,
    traj_name: str,
    gmx_command: str = GMX_COMMAND,
    prev_gro_name: str = PREV_GRO_NAME,
    prev_gro_dir: str = PREV_GRO_DIR,
    traj_override: str | list[str] | None = None,
    combine_repeats: bool = True,
    dist_cutoff_nm: float = 0.35,
    angle_cutoff_deg: float = 135.0,
) -> np.ndarray:
    """
    Calculate hydrogen-bond contacts between the peptide `pdb` and the given
    ligand group (`group` on `ligand_resname`).

    This function:
        1) Builds the dist/angle index.
        2) Runs GROMACS distance/angle for each repeat (respecting traj_override).
        3) Reads all XVG files into DataFrames.
        4) Applies distance/angle cutoffs to build a boolean contact array.

    Returns
    -------
    contacts : np.ndarray
        Boolean array of shape (3 * len(pdb), n_frames).
        For residue i (0-based):
            row 3*i     : Backbone amine/amide contacts
            row 3*i + 1 : Side-chain contacts
            row 3*i + 2 : Backbone carbonyl/carboxyl contacts
    """
    # 1) Run GROMACS and get file paths + headers
    dist_files, angle_files, dist_header, angle_header = run_gmx_contacts_for_group(
        pdb=pdb,
        ligand=ligand,
        ligand_resname=ligand_resname,
        group=group,
        master_dir=master_dir,
        job_name=job_name,
        number_repeats=number_repeats,
        traj_name=traj_name,
        gmx_command=gmx_command,
        prev_gro_name=prev_gro_name,
        prev_gro_dir=prev_gro_dir,
        traj_override=traj_override,
        combine_repeats=combine_repeats,
    )

    # 2) Read numerical data
    dist_df = read_contact_distances_xvg(dist_files, dist_header)
    angle_df = read_contact_angles_xvg(angle_files, angle_header)

    if dist_df.empty or angle_df.empty:
        print(
            f"[WARN] No contact data for {pdb} {ligand_resname} {group}; "
            "returning empty contact matrix."
        )
        return np.zeros((3 * len(pdb), 0), dtype=bool)

    n_frames = len(dist_df["time"])
    contacts = np.zeros((3 * len(pdb), n_frames), dtype=bool)

    # 3) Apply HB definitions (same logic as original calc_contacts)
    for i in range(len(pdb)):
        res_one = pdb[i]
        res_three = aa_abbr[res_one]

        # --- Backbone amine/amide (row 3*i) ---
        temp = np.zeros(n_frames, dtype=bool)

        if i == 0:
            if res_three == "PRO":
                backbone_donors = aa_b["Imine"]["D"]
            else:
                backbone_donors = aa_b["Amine"]["D"]
        else:
            if res_three == "PRO":
                backbone_donors = aa_b["Imide"]["D"]
            else:
                backbone_donors = aa_b["Amide"]["D"]

        for d in backbone_donors:
            for a in hb_lig[ligand_resname][group]["A"]:
                dist_col = f"res {i+1} {d.split()[0]} {a}"
                angle_col = f"res {i+1} {d} {a}"
                if dist_col in dist_df.columns and angle_col in angle_df.columns:
                    hbonds = (dist_df[dist_col] <= dist_cutoff_nm) & (
                        angle_df[angle_col] >= angle_cutoff_deg
                    )
                    temp |= hbonds

        contacts[i * 3] = temp

        # --- Side chain (row 3*i + 1) ---
        temp = np.zeros(n_frames, dtype=bool)

        # Peptide side-chain donor -> ligand acceptor
        for d in aa_s[res_three]["D"]:
            for a in hb_lig[ligand_resname][group]["A"]:
                dist_col = f"res {i+1} {d.split()[0]} {a}"
                angle_col = f"res {i+1} {d} {a}"
                if dist_col in dist_df.columns and angle_col in angle_df.columns:
                    hbonds = (dist_df[dist_col] <= dist_cutoff_nm) & (
                        angle_df[angle_col] >= angle_cutoff_deg
                    )
                    temp |= hbonds

        # Ligand donor -> peptide side-chain acceptor
        for a in aa_s[res_three]["A"]:
            for d in hb_lig[ligand_resname][group]["D"]:
                dist_col = f"res {i+1} {d.split()[0]} {a}"
                angle_col = f"res {i+1} {d} {a}"
                if dist_col in dist_df.columns and angle_col in angle_df.columns:
                    hbonds = (dist_df[dist_col] <= dist_cutoff_nm) & (
                        angle_df[angle_col] >= angle_cutoff_deg
                    )
                    temp |= hbonds

        contacts[i * 3 + 1] = temp

        # --- Backbone carbonyl/carboxyl (row 3*i + 2) ---
        temp = np.zeros(n_frames, dtype=bool)

        if i == len(pdb) - 1:
            backbone_acceptors = aa_b["Carboxyl"]["A"]
        else:
            backbone_acceptors = aa_b["Carbonyl"]["A"]

        for a in backbone_acceptors:
            for d in hb_lig[ligand_resname][group]["D"]:
                dist_col = f"res {i+1} {d.split()[0]} {a}"
                angle_col = f"res {i+1} {d} {a}"
                if dist_col in dist_df.columns and angle_col in angle_df.columns:
                    hbonds = (dist_df[dist_col] <= dist_cutoff_nm) & (
                        angle_df[angle_col] >= angle_cutoff_deg
                    )
                    temp |= hbonds

        contacts[i * 3 + 2] = temp

    return contacts

def summarize_contacts_to_fractions(
    contacts: np.ndarray,
    pdb: str,
) -> np.ndarray:
    """
    Convert a (3*len(pdb), n_frames) boolean contact matrix into
    per-residue, per-group fractions.

    Output shape: (len(pdb), 3), with columns:
        [backbone_NH_frac, sidechain_frac, backbone_CO_frac]
    """
    if contacts.size == 0:
        return np.zeros((len(pdb), 3), dtype=float)

    n_res = len(pdb)
    n_frames = contacts.shape[1]
    if contacts.shape[0] != 3 * n_res:
        raise ValueError(
            f"[summarize_contacts_to_fractions] Expected {3*n_res} rows, "
            f"got {contacts.shape[0]}"
        )

    frac = np.zeros((n_res, 3), dtype=float)
    for i in range(n_res):
        for j in range(3):
            row = 3 * i + j
            frac[i, j] = contacts[row].sum() / max(n_frames, 1)

    return frac


def plot_contacts_heatmap(
    contact_fractions: np.ndarray,
    pdb: str,
    ligand: str,
    ligand_resname: str,
    group: str,
    master_dir: str,
    out_png: str | None = None,
    cutoff = None
):
    """
    Plot a simple heatmap: residues (y) vs 3 groups (x):
        backbone NH, side chain, backbone CO

    Values are fractions in [0, 1].
    """
    if contact_fractions.size == 0:
        print(
            f"[WARN] Empty contact fractions for {pdb} {ligand_resname} {group}; not plotting."
        )
        return

    md, analysis_dir = proj_paths(master_dir)
    ensure_dir(analysis_dir)

    n_res = contact_fractions.shape[0]
    res_labels = [f"{i+1}" for i in range(n_res)]
    group_labels = ["BB-NH/amide", "Side chain", "BB-CO/COOH"]

    fig, ax = plt.subplots(figsize=(4, 0.4 * n_res + 1), dpi=300)
    im = sns.heatmap(
        contact_fractions,
        ax=ax,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Contact fraction"},
        xticklabels=group_labels,
        yticklabels=res_labels,
    )
    ax.set_xlabel("Group")
    ax.set_ylabel("Residue index")

    ligand_str = get_ligand_string(ligand)
    title = f"{pdb.upper()} {ligand_str} – {ligand_resname} {group}"
    ax.set_title(title)

    fig.tight_layout()
    if out_png is None:
        if cutoff is None:
            fname = f"contacts_{pdb}_{ligand}_{ligand_resname}_{group}.png"
        else:
            fname= f"contacts_{pdb}_{ligand}_{group}_c{cutoff}.png"
    out_png = os.path.join(
            analysis_dir, fname
        )
    fig.savefig(out_png, dpi=300)
    print(f"[SAVED] Contacts heatmap -> {out_png}")
    plt.close(fig)

def contacts_run(
    pdb: str,
    ligand: str,
    master_dir: str,
    job_name: str = JOB_NAME,
    number_repeats: int = NUMBER_REPEATS,
    traj_name: str = TRAJ_NAME,
    gmx_command: str = GMX_COMMAND,
    prev_gro_name: str = PREV_GRO_NAME,
    prev_gro_dir: str = PREV_GRO_DIR,
    traj_override: str | list[str] | None = None,
    combine_repeats: bool = True,
    cutoff = None
) -> dict[str, np.ndarray]:
    """
    High-level driver for HB contacts, in the same style as rmsd().

    For a given (pdb, ligand), this:
        - Maps `ligand` -> ligand_resname (e.g. 'hpo4' -> 'PO4').
        - Chooses appropriate ligand groups to analyze.
        - Runs contact calculation for each group (respecting traj_override).
        - Plots a heatmap for each group.
        - Returns a dict {group: fractions}, where fractions is (n_res, 3).
    """
    ligand_resname = system_ligand_to_resname(ligand)
    if ligand_resname is None:
        print(f"[INFO] contacts_run: ligand={ligand} (unliganded); skipping contacts.")
        return {}

    # Decide which ligand groups to analyze, based on residue name
    if ligand_resname == "PO4":
        group_list = ["Phos"]
    elif ligand_resname == "GTP":
        # You can trim/adjust this set depending on what you actually want
        group_list = ["a", "b", "g", "base", "ribose"]
    elif ligand_resname == "ATP":
        group_list = ["a", "b", "g", "base", "ribose"]
    else:
        raise ValueError(
            f"[contacts_run] No group list defined for ligand_resname={ligand_resname}"
        )

    results: dict[str, np.ndarray] = {}
    for group in group_list:
        print(f"[INFO] contacts_run: {pdb} {ligand_resname} group={group}")
        contacts = calc_contacts_for_group(
            pdb=pdb,
            ligand=ligand,
            ligand_resname=ligand_resname,
            group=group,
            master_dir=master_dir,
            job_name=job_name,
            number_repeats=number_repeats,
            traj_name=traj_name,
            gmx_command=gmx_command,
            prev_gro_name=prev_gro_name,
            prev_gro_dir=prev_gro_dir,
            traj_override=traj_override,
            combine_repeats=combine_repeats,
        )
        fractions = summarize_contacts_to_fractions(contacts, pdb)
        results[group] = fractions

        plot_contacts_heatmap(
            contact_fractions=fractions,
            pdb=pdb,
            ligand=ligand,
            ligand_resname=ligand_resname,
            group=group,
            master_dir=master_dir,
            cutoff=cutoff
        )

    return results

#------------Cleanup
def move_temp_xtc_and_cleanup(work_dir):
    tmp_dir = os.path.join(work_dir, "delete")
    ensure_dir(tmp_dir)

    for name in os.listdir(work_dir):
        if name.endswith(".xtc") or name.startswith("#angdist"):
            try:
                shutil.move(os.path.join(work_dir, name), os.path.join(tmp_dir, name))
            except Exception:
                pass

    shutil.rmtree(tmp_dir, ignore_errors=True)

def run_full_pipeline(
        ligands = LIGANDS,
        pdbs = PDBS,
        master_dir = None,
        cutoff = None,
        number_repeats = NUMBER_REPEATS,
        traj_name = TRAJ_NAME,
        gmx_command = GMX_COMMAND,
        job_name = JOB_NAME,
        prev_gro_name = PREV_GRO_NAME,
        do_plots = True,
        do_cleanup = True):

    md, analysis_dir = proj_paths(master_dir)
    #all data is returned as dataframe for optional evaluation
    all_dist= []   
    all_angle = []
    all_rog = []
    all_contacts = []

    for ligand in ligands:
        for pdb in pdbs:
            df_dist, df_angle = dist_and_angle_run(
                    pdb=pdb,
                    ligand=ligand,
                    master_dir=md,
                    cutoff = cutoff,
                    number_repeats=number_repeats,
                    traj_name = traj_name,
                    gmx_command = gmx_command,
                    job_name = job_name,
                    analysis_dir = analysis_dir,
                    do_plots = do_plots
                    )
            all_dist.append(df_dist)
            all_angle.append(df_angle)
            
            if do_cleanup:
                move_temp_xtc_and_cleanup(md)

            dihedral_nestiness_figure(
                    pdb,
                    ligand,
                    md,
                    res1=4,
                    res2=7,
                    prev_gro_name=prev_gro_name,
                    job_name = job_name,
                    number_repeats=number_repeats,
                    out_png = os.path.join(analysis_dir, f"nest_classify_{pdb}_{ligand}.png")
            )

            rog_results = rog_run(
                    pdb=pdb,
                    ligand=ligand,
                    master_dir=md,
                    analysis_dir=analysis_dir,
                    number_repeats=number_repeats,
                    traj_name = None,
                    out_png = None,
                    )
            all_rog.append(rog_results)

            process_abego(master_dir = md,
                    job_name = job_name,
                    number_repeats = number_repeats,
                    traj_name = traj_name,
                    indexing_pairs = INDEXING_PAIRS,
                    cutoff = None,
                    labels = None
                    )

            rmsd(ligand_name=ligand,
                pdb=pdb,
                master_directory = md,
                job_name=job_name,
                number_of_repeats=number_of_repeats,
                )

            contacts_results = contacts_run(
                pdb=pdb,
                ligand=ligand,
                master_dir=md,
                job_name=job_name,
                number_repeats=number_repeats,
                traj_name=traj_name,
                gmx_command=gmx_command,
                prev_gro_name=prev_gro_name,
                prev_gro_dir=PREV_GRO_DIR,
                traj_override=None,          # or adsorption trajs later
                combine_repeats=True,
            )

            all_contacts.append((pdb, ligand, contacts_results))


    df_dist_all = pd.concat(all_dist, ignore_index=True) if all_dist else pd.DataFrame()
    df_angle_all = pd.concat(all_angle, ignore_index=True) if all_angle else pd.DataFrame()
    df_rog_all = pd.concat(all_rog, ignore_index=True) if all_rog else pd.DataFrame() #MAY ERROR

    return df_dist_all, df_angle_all, df_rog_all

if __name__ == "__main__":
    try:
        dist_df, angle_df = run_full_pipeline(
            ligands = LIGANDS,
            pdbs = PDBS,
            master_dir=None,
            number_repeats = NUMBER_REPEATS,
            traj_name = TRAJ_NAME,
            gmx_command = GMX_COMMAND,
            job_name = JOB_NAME,
            do_plots = True,
            do_cleanup=True)
        print("Done. DF shapes:", dist_df.shape, angle_df.shape)
    except Exception as e:
        print(f"[ERROR] {e}")

import json
from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from tmd.constants import DEFAULT_ATOM_MAPPING_KWARGS
from tmd.fe.atom_mapping import get_cores_and_diagnostics, get_num_dummy_atoms
from tmd.fe.utils import get_mol_name, read_sdf

STAR_MAP = "star_map"
GREEDY = "greedy"
BEST = "best"

JACCARD = "jaccard"
DUMMY_ATOMS = "dummy_atoms"

CORE_FIELD = "core"
MOL_FIELD = "mol"


def atom_mapping_jaccard_distance(mol_a, mol_b, core) -> float:
    return 1 - (len(core) / (mol_a.GetNumAtoms() + mol_b.GetNumAtoms() - len(core)))


def refine_mapping_wrapper(args):
    try:
        mol_a, mol_b, atom_mapping_kwargs = args
        assert atom_mapping_kwargs["initial_mapping"] is not None
        cores, _ = get_cores_and_diagnostics(mol_a, mol_b, **atom_mapping_kwargs)
        core = cores[0]
    except Exception:
        core = atom_mapping_kwargs["initial_mapping"]
    return core


def atom_mapping_wrapper(pair, atom_mapping_kwargs=DEFAULT_ATOM_MAPPING_KWARGS.copy()):
    try:
        idx, mol_a, mol_b = pair
        cores, diag = get_cores_and_diagnostics(mol_a, mol_b, **atom_mapping_kwargs)
        core = cores[0]
    except Exception:
        core = None
        diag = None
    return idx, core, diag


def generate_nxn_atom_mappings(
    mols: list, atom_mapping_kwargs: dict[str, Any] = DEFAULT_ATOM_MAPPING_KWARGS.copy()
) -> NDArray:
    all_pairs = [((i, j), mols[i], mols[j]) for i in range(len(mols)) for j in range(i + 1, len(mols))]
    core_matrix = np.empty((len(mols), len(mols)), dtype=object)
    with Pool() as pool:
        for res in pool.imap_unordered(
            partial(atom_mapping_wrapper, atom_mapping_kwargs=atom_mapping_kwargs), all_pairs
        ):
            (mol_a_idx, mol_b_idx), core, _ = res
            core_matrix[mol_a_idx, mol_b_idx] = core
    return core_matrix


def build_star_graph(
    hub_cmpd_name: str, mols: list, atom_mapping_kwargs: dict[str, Any] = DEFAULT_ATOM_MAPPING_KWARGS.copy()
):
    mols_by_name = {get_mol_name(m): m for m in mols}
    assert hub_cmpd_name in mols_by_name
    hub_cmpd = mols_by_name[hub_cmpd_name]

    graph = nx.DiGraph()
    for mol in mols:
        graph.add_node(get_mol_name(mol), **{MOL_FIELD: mol})

    i = 0
    pairs = []
    for name, mol in mols_by_name.items():
        if name == hub_cmpd_name:
            continue
        pairs.append((i, hub_cmpd, mol))
        i += 1
    with Pool() as pool:
        for res in pool.imap_unordered(partial(atom_mapping_wrapper, atom_mapping_kwargs=atom_mapping_kwargs), pairs):
            i, core, _ = res
            graph.add_edge(hub_cmpd_name, get_mol_name(pairs[i][2]), **{CORE_FIELD: core})
    return graph


def build_greedy_graph(mols, scoring_methods: list[str], k_min_cut: int = 2) -> nx.DiGraph:
    """Build a densely connected graph using a greedy method

    Parameters
    ----------
    mols:
        List of Rdkit mols

    scoring_methods: list of string
        List of name of scoring methods to use. Will return the graph with the fewest number of dummy atoms

    k_min_cut: int
        Number of edges that can be cut before producing a disconnected graph. See networkx.k_edge_augmentation for more details

    Returns
    -------
        networkx.DiGraph
    """
    assert k_min_cut >= 1
    assert len(scoring_methods) >= 1
    mol_name_to_idx = {get_mol_name(m): i for i, m in enumerate(mols)}
    core_matrix = generate_nxn_atom_mappings(mols)

    def count_graph_dummy_atoms(g):
        return np.sum([data[DUMMY_ATOMS] for _, _, data in g.edges(data=True)])

    best_graph = None
    # Try both scoring methods
    for score_method in scoring_methods:
        possible_edges = []
        for i in range(len(mols)):
            for j in range(i + 1, len(mols)):
                core = core_matrix[i, j]
                if core is None:
                    continue
                mol_a = mols[i]
                mol_b = mols[j]

                if score_method == JACCARD:
                    edge_score = atom_mapping_jaccard_distance(mol_a, mol_b, core)
                elif score_method == DUMMY_ATOMS:
                    edge_score = float(get_num_dummy_atoms(mol_a, mol_b, core))
                else:
                    assert 0, f"Invalid score_fn: {score_method}"
                possible_edges.append((get_mol_name(mol_a), get_mol_name(mol_b), edge_score))

        graph = nx.DiGraph()
        for mol in mols:
            graph.add_node(get_mol_name(mol), **{MOL_FIELD: mol})

        for a, b in nx.k_edge_augmentation(graph.to_undirected(as_view=True), k_min_cut, possible_edges, partial=True):
            (i, j) = sorted([mol_name_to_idx[a], mol_name_to_idx[b]])
            core = core_matrix[i, j]
            src_mol = get_mol_name(mols[i])
            dst_mol = get_mol_name(mols[j])
            dummy_atoms = float(get_num_dummy_atoms(mols[i], mols[j], core))
            graph.add_edge(src_mol, dst_mol, **{CORE_FIELD: core, DUMMY_ATOMS: dummy_atoms})
        if best_graph is None:
            best_graph = graph
        elif count_graph_dummy_atoms(best_graph) > count_graph_dummy_atoms(graph):
            best_graph = graph
    assert best_graph is not None
    return best_graph


def refine_atom_mapping(nx_graph, cutoff: float):
    edges = []
    for a, b, data in nx_graph.edges(data=True):
        if CORE_FIELD not in data:
            continue
        new_kwargs = DEFAULT_ATOM_MAPPING_KWARGS.copy()
        new_kwargs["initial_mapping"] = data[CORE_FIELD]
        new_kwargs["ring_cutoff"] = cutoff
        new_kwargs["chain_cutoff"] = cutoff
        edges.append((nx_graph.nodes[a][MOL_FIELD], nx_graph.nodes[b][MOL_FIELD], new_kwargs))
    if len(edges) == 0:
        return nx_graph
    with Pool() as pool:
        refined_cores = pool.map(refine_mapping_wrapper, edges)
    refined_graph = nx_graph.copy()
    refined_graph.update(
        [(get_mol_name(a), get_mol_name(b), {CORE_FIELD: core}) for (a, b, _), core in zip(edges, refined_cores)]
    )
    return refined_graph


def main():
    parser = ArgumentParser()
    parser.add_argument("ligands_sdf")
    parser.add_argument("output_path", help="Json file to write out containing all of the edges")
    parser.add_argument(
        "--mode", default=GREEDY, choices=[STAR_MAP, GREEDY], help="Whether to generate a star map or a greedy map"
    )
    parser.add_argument("--hub_cmpd", help="If generating a star map, provide a hub compound")
    parser.add_argument(
        "--refine_cutoff",
        type=float,
        help="Refine the final graph with a new atom map cutoff, uses the initial atom mapping.",
    )
    parser.add_argument(
        "--greedy_scoring",
        choices=[BEST, JACCARD, DUMMY_ATOMS],
        default=BEST,
        help=f"How to score edges when generating greedy maps. The {BEST} option will try multiple scoring functions and return the mapping with the fewest dummy atoms",
    )

    parser.add_argument(
        "--greedy_k_min_cut",
        default=2,
        type=int,
        help="K min cut of graph to generate, only applicable if using greedy map generation",
    )
    parser.add_argument(
        "--report_interval", default=100, type=int, help="How often to report pairs, only relevant to greedy"
    )
    parser.add_argument("--ligands", nargs="+", default=None, help="Name of ligands to consider")
    parser.add_argument("--verbose", action="store_true", help="Report information about the dataset")
    args = parser.parse_args()

    np.random.seed(2025)

    assert args.output_path.endswith(".json")

    ligand_path = Path(args.ligands_sdf).expanduser()

    mols = read_sdf(ligand_path)
    mols_by_name = {get_mol_name(m): m for m in mols}
    if args.ligands is not None and len(args.ligands):
        mols = [mol for name, mol in mols_by_name.items() if name in args.ligands]
    if args.mode == GREEDY:
        if args.greedy_scoring == BEST:
            scoring_methods = [JACCARD, DUMMY_ATOMS]
        else:
            scoring_methods = [args.greedy_scoring]
        nx_graph = build_greedy_graph(mols, scoring_methods, k_min_cut=args.greedy_k_min_cut)
    else:
        assert args.hub_cmpd is not None
        nx_graph = build_star_graph(args.hub_cmpd, mols)

    if args.refine_cutoff is not None:
        nx_graph = refine_atom_mapping(nx_graph, args.refine_cutoff)

    json_output = []
    # Sort the edges to ensure determinism
    for a, b, data in sorted(nx_graph.edges(data=True), key=lambda x: f"{x[0]}_{x[1]}"):
        edge = {"mol_a": a, "mol_b": b}
        if CORE_FIELD in data:
            edge[CORE_FIELD] = data[CORE_FIELD].tolist()
        json_output.append(edge)
    print(f"Generated {args.mode} map with {len(json_output)} edges")
    if args.verbose:
        dummy_atoms = [
            get_num_dummy_atoms(mols_by_name[a], mols_by_name[b], data[CORE_FIELD])
            for a, b, data in nx_graph.edges(data=True)
            if CORE_FIELD in data
        ]
        print("Total Dummy Atoms", sum(dummy_atoms))
        print("Mean Dummy Atoms", np.round(np.mean(dummy_atoms), 2))
        print("Median Dummy Atoms", np.round(np.median(dummy_atoms), 2))
        print("Min Dummy Atoms", np.min(dummy_atoms))
        print("Max Dummy Atoms", np.max(dummy_atoms))
    with open(Path(args.output_path).expanduser(), "w") as ofs:
        json.dump(json_output, ofs, indent=1)


if __name__ == "__main__":
    main()

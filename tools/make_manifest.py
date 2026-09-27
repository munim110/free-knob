"""Write MANIFEST.md: every released artefact with its hash, producer and consumer.

Run from the repository root:

    python tools/make_manifest.py

The hashes let a reader confirm that the files they downloaded are the files the
paper was built from. The producer column says which script writes each file and
the "backs" column says which paper quantities depend on it.
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402

# artefact glob -> (producing script, paper quantities it backs)
PROVENANCE = [
    ("freeknob_parity_*.json", "tests/sevir_parity.py",
     "30-cell real-data parity of the installable FreeKnob implementation"),
    ("sevir_poolingq_*.json", "sevir/pooling_control.py",
     "Six*, Dose*, Eight*, Contrast* macros; fig_mechanism; tab_pooling"),
    ("sevir_published_contrasts.json", "sevir/published_contrasts.py",
     "Contrast*, Defl*, SameArch*, EFCasc* macros; tab_contrasts; tab_samearch"),
    ("sevir_published_contrasts.md", "sevir/published_contrasts.py",
     "human-readable form of the above, not read by the build"),
    ("sevir_decomposition.md", "sevir/decomposition_report.py",
     "human-readable form of the CasCast decomposition, not read by the build"),
    ("leadtime/*.json", "sevir/leadtime_qmap.py",
     "Lead* macros; tab_lead; Appendix on the map fitted per lead time"),
    ("knobrule/*.json", "sevir/knob_selection.py",
     "Knob* macros; tab_knobsevir; the family-against-rule appendix"),
    ("sevir_bootstrap_cascast.json", "sevir/bootstrap.py",
     "Boot* macros; tab_boot"),
    ("sevir_contrast_bootstrap.json", "sevir/contrast_bootstrap.py",
     "CBoot* macros: clustered intervals on the contrast correlation and the "
     "reversal count"),
    ("sevir_perevent_counts.npz", "sevir/contrast_bootstrap.py",
     "per-event contingency counts, so the bootstrap reruns without the "
     "prediction HDF5s"),
    ("sevir_schematic.npz", "sevir/make_schematic.py",
     "the fields behind Figure 1(a)"),
    ("sevir_val_fitted.json", "sevir/val_fitted_control.py",
     "Val* macros; tab_valfit"),
    ("sevir_val_fitted.md", "sevir/val_fitted_control.py",
     "human-readable form of the above, not read by the build"),
    ("sevir_base_rates.json", "sevir/base_rates.py",
     "RareTop, RareTopOne"),
    ("atmospheric_base_rates.json", "atmospheric/base_rates.py",
     "all atmospheric base-rate definitions, split spans and evaluated cohorts"),
    ("calibration_sensitivity_vs_rarity.md", "sevir/rarity_sweep.py",
     "Eone* macros; tab_rarity"),
    ("crowd_pooling_s*.json", "crowd/pooling_control.py",
     "Crowd* macros; tab_crowdpool"),
    ("crowd_pooling_report.md", "crowd/pooling_report.py",
     "human-readable aggregation over seeds, not read by the build"),
    ("crowd_matched_report.md", "crowd/matched_report.py",
     "CrowdMatched, CrowdSwing; tab_crowd"),
    ("radar_matched_report.md", "radar/matched_report.py",
     "Radar* macros; tab_radar"),
    ("seg_pooling.json", "segmentation/pooling_control.py",
     "Seg* macros; tab_seg"),
    ("seg_repro.json", "segmentation/repro_check.py",
     "SegRepro* macros"),
    ("selection_matched_csi.md", "selection/protocol_selection_report.py",
     "AtmosBehind, AtmosTotal"),
    ("selection_matched_csi.json", "selection/protocol_selection_report.py",
     "machine-readable form of the above, written alongside it"),
    # Intermediate artefacts. These are inputs to the reports above rather than
    # things the paper reads directly. They are shipped so that every analysis
    # stage runs without retraining, which is the expensive part.
    ("rsweep_control_matched_t*.json", "atmospheric/calibration_control.py",
     "input to sevir/rarity_sweep.py and selection/protocol_selection_report.py"),
    ("rsweep_R_table.json", "atmospheric/base_rates.py",
     "all-frame test positive fractions on the evaluation footprint, input to "
     "the rarity sweep and selection report"),
    ("R_clim_train_B08.json", "atmospheric/base_rates.py",
     "eventful-frame training climatology on the evaluation footprint, input "
     "to selection rule D"),
    ("allscene_pooled/rsweep_control_allscene_t*.json",
     "atmospheric/calibration_control.py",
     "all-scene pooled atmospheric sensitivity, with no coverage filter"),
    ("allscene_pooled/atmospheric_rarity_allscene_pooled.md",
     "sevir/rarity_sweep.py",
     "human-readable all-scene pooled rarity sensitivity"),
    ("allscene_pooled/atmospheric_selection_allscene_pooled.md",
     "selection/protocol_selection_report.py",
     "human-readable all-scene pooled matched-arm sensitivity"),
    ("allscene_pooled/atmospheric_selection_allscene_pooled.json",
     "selection/protocol_selection_report.py",
     "machine-readable all-scene pooled matched-arm sensitivity"),
    ("posthoc_crowd_matched.json", "crowd/calibration_control.py",
     "input to crowd/matched_report.py"),
    ("posthoc_radar_dual_decoder_matched_s42.json", "radar/calibration_control.py",
     "input to radar/matched_report.py"),
    ("posthoc_radar_plain_mse_s4*.json", "radar/calibration_control.py",
     "input to radar/matched_report.py, three seeds"),
]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    res = P.RESULTS
    rows, seen = [], set()
    for pattern, producer, backs in PROVENANCE:
        for f in sorted(res.glob(pattern)):
            rows.append((f.name, f.stat().st_size, sha256(f), producer, backs))
            seen.add(f.name)

    stray = sorted(f.name for f in res.iterdir()
                   if f.is_file() and f.name not in seen)

    L = ["# Manifest", "",
         "Generated by `tools/make_manifest.py`. Every file listed here is read by "
         "the build described in REPRODUCE.md, or is a human-readable rendering of "
         "a file that is.", "",
         f"{len(rows)} artefacts, "
         f"{sum(r[1] for r in rows) / 1e6:.1f} MB total.", "",
         "| artefact | bytes | sha256 (first 16) | produced by | backs |",
         "|---|---|---|---|---|"]
    for name, size, h, producer, backs in rows:
        L.append(f"| `{name}` | {size:,} | `{h[:16]}` | `{producer}` | {backs} |")

    if stray:
        L += ["", "## Present but not part of the build", ""]
        L += [f"- `{s}`" for s in stray]

    L += ["", "## Verifying", "",
          "```", "python tools/make_manifest.py", "git diff --exit-code MANIFEST.md", "```", "",
          "A clean diff means the artefacts on disk are the ones this manifest "
          "was written against."]

    out = P.ARTIFACT / "MANIFEST.md"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}: {len(rows)} artefacts, {len(stray)} not part of the build")


if __name__ == "__main__":
    main()

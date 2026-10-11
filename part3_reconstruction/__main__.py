"""Run `python -m part3_reconstruction --help` from the repository root."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from . import DataLoader, ImagingData, ImageGenerator, VolumeReconstructor
from .validation import dice_score


def main():
    parser = argparse.ArgumentParser(description="LFI imaging: schema inspection, UQ preview and synthetic validation")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="List HDF5 dataset schemas without loading waveforms")
    inspect.add_argument("path")
    reconstruct = sub.add_parser("reconstruct", help="Build a calibrated sequence from an explicit JSON manifest")
    reconstruct.add_argument("manifest")
    reconstruct.add_argument("--output", required=True)
    reconstruct.add_argument("--spacing", type=float)
    reconstruct.add_argument("--max-voxels", type=int, default=50_000_000)
    audit = sub.add_parser("audit", help="Audit all HDF5 image maps in a local directory")
    audit.add_argument("directory")
    audit.add_argument("--output", default="outputs/audit.json")
    preview = sub.add_parser("uq-preview", help="Render observed UQ frame HDF5 maps without guessing phase calibration")
    preview.add_argument("path")
    preview.add_argument("--side", choices=["left", "right"], default="left")
    preview.add_argument("--phase-scale", type=float)
    preview.add_argument("--output", default="outputs/real")
    bench = sub.add_parser("benchmark", help="Time and peak memory for 2D rendering, resampling and export")
    bench.add_argument("files", nargs="*", help="Optional UQ frame HDF5 files")
    bench.add_argument("--output", default="outputs/benchmark")
    demo = sub.add_parser("demo", help="Synthetic validation only; not experimental evidence")
    demo.add_argument("--output", default="outputs/synthetic")
    args = parser.parse_args()
    if args.command == "reconstruct":
        from .sequence import reconstruct_manifest
        print(json.dumps(reconstruct_manifest(args.manifest, args.output, spacing=args.spacing,
                                              max_voxels=args.max_voxels), indent=2))
        return
    if args.command == "benchmark":
        from .benchmark import run
        out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
        report = run(real_files=args.files, out_dir=out)
        (out / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    if args.command == "audit":
        from .audit import audit_directory
        report = audit_directory(args.directory, args.output)
        print(json.dumps(report["summary"], indent=2))
        if report["summary"]["errors"]:
            raise SystemExit(2)
        return
    if args.command == "inspect":
        print(json.dumps(DataLoader.inspect_hdf5(args.path), indent=2))
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.command == "uq-preview":
        data = DataLoader.load_uq_frame(args.path, side=args.side, phase_scale=args.phase_scale)
        stem = Path(args.path).stem + "_" + args.side
        generator = ImageGenerator(data)
        generator.generate_amplitude(export=output / (stem + "_amplitude.png"))
        generator.generate_phase("sequential", export=output / (stem + "_phase.png"))
        generator.generate_comparison(phase_mode="sequential", export=output / (stem + "_comparison.png"))
        digest = hashlib.sha256()
        with open(args.path, "rb") as f:
            for chunk in iter(lambda: f.read(1024*1024), b""):
                digest.update(chunk)
        stats = {"source_sha256": digest.hexdigest(), "metadata": data.metadata,
                 "shape_yx": list(data.amplitude.shape), "validation_scope": "I/O and rendering smoke test only; no calibrated phase or reference ground truth"}
        for name in ("amplitude", "phase"):
            a = getattr(data, name)
            stats[name] = {"minimum": float(np.nanmin(a)), "maximum": float(np.nanmax(a)),
                           "finite_fraction": float(np.isfinite(a).mean()),
                           "percentiles_2_98": np.nanpercentile(a, [2, 98]).tolist()}
        (output / (stem + "_report.json")).write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(json.dumps(stats, indent=2))
    else:
        z = np.arange(-10, 11, 2, dtype=float)
        zz, yy, xx = np.meshgrid(z, np.arange(-10, 11), np.arange(-10, 11), indexing="ij")
        amplitude = np.exp(-(xx*xx+yy*yy+zz*zz)/30)
        data = ImagingData(amplitude, .8*xx + .3*yy + zz, {"phase_state": "unwrapped", "source": "analytic synthetic Gaussian + linear phase", "spacing_xy": (1, 1), "spatial_unit": "mm", "origin_xy": (-10, -10)})
        ImageGenerator(data.slice(5)).generate_comparison(export=output / "comparison.png")
        ImageGenerator(data.slice(5)).generate_comparison("hsl_fusion", export=output / "fusion.png")
        v = VolumeReconstructor(data, spacing_xy=(1, 1), z_coordinates=z, origin_xy=(-10, -10)).resample_isotropic(1)
        v.export_nifti(output / "amplitude.nii.gz")
        v.export_nifti(output / "phase.nii.gz", "phase")
        sphere = ImagingData((xx*xx+yy*yy+zz*zz <= 49).astype(float), zz, {"phase_state": "unwrapped"})
        sv = VolumeReconstructor(sphere, spacing_xy=(1, 1), z_coordinates=z).resample_isotropic(1)
        tz, ty, tx = np.meshgrid(np.arange(-10, 11), np.arange(-10, 11), np.arange(-10, 11), indexing="ij")
        report = {"kind": "synthetic validation only", "volume_shape_zyx": list(v.data.amplitude.shape),
                  "linear_phase_max_abs_error": float(np.max(np.abs(v.data.phase-(.8*tx+.3*ty+tz)))),
                  "coarsely_sampled_binary_sphere_dice": dice_score(sv.data.amplitude >= .5, tx*tx+ty*ty+tz*tz <= 49),
                  "limitation": "trilinear interpolation cannot recover binary boundary detail lost at acquisition"}
        (output / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

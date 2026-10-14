from part3_reconstruction.benchmark import run


def test_benchmark_small(tmp_path):
    r = run(shapes=((3, 16, 16),), out_dir=tmp_path)
    row = r["results"][0]
    assert row["resampled_shape_zyx"] == [5, 16, 16]
    for key in ("render_2d_comparison", "resample_to_1mm", "export_nifti"):
        assert row[key]["seconds"] >= 0 and row[key]["peak_traced_mb"] >= 0
    assert (tmp_path / "bench_3x16x16.nii.gz").exists()

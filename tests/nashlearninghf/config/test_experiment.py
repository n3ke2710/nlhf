from nashlearninghf.config.experiment import ExperimentConfig


def test_experiment_config_creates_dirs(tmp_path):
    out = tmp_path / "out"
    log = tmp_path / "logs"
    gfx = tmp_path / "gfx"

    config = ExperimentConfig(output_dir=str(out), log_dir=str(log), graphics_dir=str(gfx))

    assert out.exists()
    assert log.exists()
    assert gfx.exists()
    assert config.output_dir == str(out)

import stat
import tarfile
from pathlib import Path

from optbench.archive_eon import adapt_optimizer_sections, run_minima
from optbench.dat import parse


def test_fire_timestep_moves_to_the_section_the_client_reads():
    text = "[Optimizer]\nopt_method=fire\ntime_step=1.5\ntime_step_max=1.5\n"
    adapted = adapt_optimizer_sections(text)
    assert "[FIRE]" in adapted
    assert "time_step = 1.5" in adapted
    assert "time_step_max = 1.5" in adapted
    assert "opt_method=fire" in adapted


def test_quickmin_timestep_keeps_the_archived_number():
    text = "[Optimizer]\nopt_method=qm\ntime_step=1.527\n"
    adapted = adapt_optimizer_sections(text)
    assert "[QuickMin]" in adapted
    assert "time_step = 1.527" in adapted


def test_neb_opt_method_is_copied_into_the_band_section():
    text = "[Nudged Elastic Band]\nimages=8\n\n[Optimizer]\nopt_method=cg\n"
    adapted = adapt_optimizer_sections(text)
    band = adapted.split("[Optimizer]")[0]
    assert "opt_method = cg" in band
    assert adapted.count("opt_method") == 2


def test_lbfgs_is_unchanged():
    text = "[Optimizer]\nopt_method=lbfgs\nconverged_force=0.01\n"
    assert adapt_optimizer_sections(text) == text


def test_neb_report_divides_by_the_image_count(tmp_path: Path):
    entry = tmp_path / "entry"
    entry.mkdir()
    (entry / "config.ini").write_text(
        "[Main]\njob = nudged_elastic_band\n\n[Optimizer]\nopt_method=lbfgs\n",
        encoding="utf-8",
    )
    (entry / "report.sh").write_text(
        "fcalls=$(awk '$2~/force_calls_neb/{if ($1<10000){fc+=$1;n+=1}}END{printf(\"%.0f\\n\", fc/n/8)}' minima/*/results.dat)\n"
        "fcalls_max=$(awk '$2~/force_calls_neb/{if ($1<10000){printf(\"%.0f\\n\", $1/8)}}' minima/*/results.dat | sort -g | tail -n 1)\n"
        "fcalls_min=$(awk '$2~/force_calls_neb/{if ($1<10000){printf(\"%.0f\\n\", $1/8)}}' minima/*/results.dat | sort -gr | tail -n 1)\n"
        "echo \"force_calls $fcalls\" > benchmark.dat\n"
        "echo \"force_calls_max $fcalls_max\" >> benchmark.dat\n"
        "echo \"force_calls_min $fcalls_min\" >> benchmark.dat\n"
        "echo \"code Eon\" >> benchmark.dat\n"
        "echo \"code_version r2090\" >> benchmark.dat\n",
        encoding="utf-8",
    )
    packed_root = tmp_path / "pack" / "minima" / "min01"
    packed_root.mkdir(parents=True)
    (packed_root / "a.con").write_text("a\n", encoding="utf-8")
    (packed_root / "b.con").write_text("b\n", encoding="utf-8")
    archive = tmp_path / "minima.tgz"
    with tarfile.open(archive, "w:gz") as packed:
        packed.add(tmp_path / "pack" / "minima", arcname="minima")
    client = tmp_path / "client"
    client.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "Path('results.dat').write_text('336 force_calls_neb\\n0 termination_reason\\n')\n",
        encoding="utf-8",
    )
    client.chmod(client.stat().st_mode | stat.S_IEXEC)
    output = tmp_path / "benchmark.dat"
    run_minima(
        entry,
        archive,
        client,
        tmp_path / "work",
        output,
        workers=1,
        code="Eon",
        version="3.5.0",
    )
    fields = parse(output.read_text(encoding="utf-8"))
    assert fields["force_calls"] == "42"
    assert fields["code_version"] == "3.5.0"
    assert fields["template"] == "entry"

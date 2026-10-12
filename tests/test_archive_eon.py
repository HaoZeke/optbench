import stat
import tarfile
import unittest
from pathlib import Path

from optbench.archive_eon import adapt_optimizer_sections, run_minima
from optbench.dat import parse

ROOT = Path(__file__).resolve().parents[1]


class ArchiveEonTests(unittest.TestCase):
    def test_fire_timestep_moves_to_the_section_the_client_reads(self):
        text = (
            "[Optimizer]\nopt_method=fire\ntime_step=1.5\ntime_step_max=1.5\n"
        )
        adapted = adapt_optimizer_sections(text)
        self.assertIn("[FIRE]", adapted)
        self.assertIn("time_step = 1.5", adapted)
        self.assertIn("time_step_max = 1.5", adapted)
        self.assertIn("opt_method=fire", adapted)

    def test_quickmin_timestep_keeps_the_archived_number(self):
        text = "[Optimizer]\nopt_method=qm\ntime_step=1.527\n"
        adapted = adapt_optimizer_sections(text)
        self.assertIn("[QuickMin]", adapted)
        self.assertIn("time_step = 1.527", adapted)

    def test_neb_opt_method_is_copied_into_the_band_section(self):
        text = (
            "[Nudged Elastic Band]\nimages=8\n\n[Optimizer]\nopt_method=cg\n"
        )
        adapted = adapt_optimizer_sections(text)
        band = adapted.split("[Optimizer]")[0]
        self.assertIn("opt_method = cg", band)
        self.assertEqual(adapted.count("opt_method"), 2)

    def test_lbfgs_is_unchanged(self):
        text = "[Optimizer]\nopt_method=lbfgs\nconverged_force=0.01\n"
        self.assertEqual(adapt_optimizer_sections(text), text if text.endswith("\n") else text + "\n")

    def test_neb_report_divides_by_the_image_count(self):
        scratch = ROOT / "build" / "neb-test"
        if scratch.exists():
            import shutil

            shutil.rmtree(scratch)
        entry = scratch / "entry"
        entry.mkdir(parents=True)
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
        packed_root = scratch / "pack" / "minima" / "min01"
        packed_root.mkdir(parents=True)
        (packed_root / "a.con").write_text("a\n", encoding="utf-8")
        (packed_root / "b.con").write_text("b\n", encoding="utf-8")
        archive = scratch / "minima.tgz"
        with tarfile.open(archive, "w:gz") as packed:
            packed.add(scratch / "pack" / "minima", arcname="minima")
        client = scratch / "client"
        client.write_text(
            "#!/usr/bin/env python3\n"
            "from pathlib import Path\n"
            "Path('results.dat').write_text('336 force_calls_neb\\n0 termination_reason\\n')\n",
            encoding="utf-8",
        )
        client.chmod(client.stat().st_mode | stat.S_IEXEC)
        output = scratch / "benchmark.dat"
        run_minima(
            entry,
            archive,
            client,
            scratch / "work",
            output,
            workers=1,
            code="Eon",
            version="3.5.0",
        )
        fields = parse(output.read_text(encoding="utf-8"))
        self.assertEqual(fields["force_calls"], "42")
        self.assertEqual(fields["code_version"], "3.5.0")
        self.assertEqual(fields["template"], "entry")


if __name__ == "__main__":
    unittest.main()

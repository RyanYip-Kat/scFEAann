import os
import subprocess
import sys


def test_root_import_does_not_load_matplotlib() -> None:
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = "/tmp/scfeaann-mpl-import-test"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, scFEAann; print('matplotlib' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.stdout.strip().splitlines()[-1] == "False"

"""Run the federated NanoGPT simulation directly (no flwr CLI needed)."""

import os

os.environ["RAY_DEDUP_LOGS"] = "0"

from pathlib import Path

from flwr.app import Context, RecordDict
from flwr.common.config import get_fused_config_from_dir
from flwr.common.typing import Run, RunStatus, UserConfig
from flwr.simulation.run_simulation import _run_simulation
from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME, NOOP_FEDERATION
from flwr.common.telemetry import EventType

from nanogpt_shakespeare.server_app import app as server_app
from nanogpt_shakespeare.client_app import app as client_app


def main():
    app_dir = Path(__file__).parent
    run_config = get_fused_config_from_dir(app_dir, {})
    print(f"Run config: {run_config}")

    run_id = 0
    run = Run.create_empty(run_id)
    run.federation = NOOP_FEDERATION
    run.override_config = run_config

    server_app_context = Context(
        run_id=run_id,
        node_id=0,
        node_config=UserConfig(),
        state=RecordDict(),
        run_config=run_config,
    )

    _run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=2,
        backend_name="ray",
        backend_config={
            "init_args": {"num_cpus": 2},
            "client_resources": {"num_cpus": 1, "num_gpus": 0},
        },
        server_app_context=server_app_context,
        run=run,
        app_dir=str(app_dir),
        verbose_logging=True,
        exit_event=EventType.PYTHON_API_RUN_SIMULATION_LEAVE,
    )


if __name__ == "__main__":
    main()

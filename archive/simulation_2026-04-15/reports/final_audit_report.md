# Final Audit Report

Date: 2026-04-15

## Scope

- Audited the hardware-facing `Viewfinder4_tk.py` path.
- Added and used a mock hardware stack for two SMUs, switch, stage, Zaber, and lights.
- Ran a full-window stress simulation with a mixed DCIV and pulse protocol.
- Rolled simulation-only files back out of the live production tree after validation.

## Audit Result

- The original audit findings are preserved in [audit_findings.md](/run/media/shreyas/Data/vipsa_clean/archive/simulation_2026-04-15/reports/audit_findings.md).
- The connection path in live `Viewfinder4_tk.py` now stays on the threaded real-hardware workflow only.
- Simulation-only GUI hooks, environment toggles, launcher script, generated artifacts, and mock backend were removed from the active runtime path and archived here.
- Artifact plot generation was hardened so background saves no longer build Matplotlib GUI figures from worker threads.

## Stress Run Result

- Status: PASS
- Full window launched and executed with mocks.
- Connect time: 0.427 s
- Grid protocol time: 12.285 s
- Total run time: 13.540 s
- Peak RSS: 792.930 MB
- Run log: [simulation_run.log](/run/media/shreyas/Data/vipsa_clean/archive/simulation_2026-04-15/runs/simulation_run.log)
- Generated artifacts: [stress_sim_artifacts](/run/media/shreyas/Data/vipsa_clean/archive/simulation_2026-04-15/runs/stress_sim_artifacts)

## Archive Layout

- Reports: [reports](/run/media/shreyas/Data/vipsa_clean/archive/simulation_2026-04-15/reports)
- Mock backend: [mock_hardware.py](/run/media/shreyas/Data/vipsa_clean/archive/simulation_2026-04-15/hardware/mock_hardware.py)
- Stress runner: [viewfinder_stress_sim.py](/run/media/shreyas/Data/vipsa_clean/archive/simulation_2026-04-15/scripts/viewfinder_stress_sim.py)
- Stress launcher: [run_viewfinder_stress_sim.sh](/run/media/shreyas/Data/vipsa_clean/archive/simulation_2026-04-15/scripts/run_viewfinder_stress_sim.sh)

## Production State

- Active runtime keeps the connection hardening, thread-safe connect and disconnect flow, and Testmaker integration.
- Active runtime no longer exposes the simulation checkbox or imports the mock backend.
- Generated caches and stray bytecode were removed during cleanup.

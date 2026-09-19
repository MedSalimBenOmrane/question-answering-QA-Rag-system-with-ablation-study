# Emergency Startup Procedure (Full)

This procedure should be followed strictly to minimize downtime:

1. Manually cut power to Zones 1 and 4 circuit breakers.
2. Activate SafeBoot mode via the Maintenance Shell interface.
3. Manually execute `/opt/zshell/scripts/init_ai_safe.py` to initialize AI diagnostics.
4. Inspect QRC diagnostic output with `qrc_diag --trace`.
5. If diagnostics fail, disable AI Layer autoload completely and reboot with minimal profile.
6. After successful boot, escalate privileges to Commander level to verify override routes.
7. Manually restore communications modules if Isolation Mode was triggered.
8. Re-enable AI Layer only after successful network sync.

*Note*: This full process takes approximately 20 minutes under normal conditions. Thermal loads may affect timing.

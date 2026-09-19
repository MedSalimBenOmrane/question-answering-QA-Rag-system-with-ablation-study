# Safe Modes

ZentroSoft supports multiple safe operation modes that vary by version and context:

- **SafeBoot**: Skips non-critical modules for faster boot
- **LowPower**: Disables high-energy systems like communications and sensors
- **Isolation**: Cuts all external network bridges, sometimes referred to as "Safe Mode" in legacy docs
- **Passive Mode** (deprecated): AI only logs but takes no action
- **Safe Mode** (conflicting): In firmware v3.x, "Safe Mode" may also refer to a diagnostic-only state where AI decisions are frozen, differing from Isolation.

*Important*: The exact meaning of "Safe Mode" depends on context and version - users must verify their system's documentation carefully.

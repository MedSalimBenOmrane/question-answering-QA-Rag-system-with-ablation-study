# Network Policies

- AI modules must not initiate outbound connections without Commander override.
- Internal telemetry uplinks are allowed.
- In "Safe Mode" (ambiguous term), external bridges may remain active or be shut down, depending on firmware version.
- v3.x firmware mistakenly allows redundant outbound packets causing leaks - fixed by Patch 7-F.
- Emergency overrides can bypass all network restrictions, but only by Commander or AI Ethics Officer.

**Contradiction**: Some docs say Isolation disables outbound traffic; others note telemetry uplinks are exempt.

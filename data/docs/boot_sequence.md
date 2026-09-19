# Boot Sequence

ZentroSoft initializes modules in this order:

1. Core Services
2. Sensor & Navigation Systems
3. AI Layer
4. User Interface Layer
5. External Network Bridge

If the QRC fails, the boot halts at phase 3.
In v4.1, SafeBoot skips the AI Layer, but this causes errors in some edge cases with mission-critical AI dependencies, fixed in 4.2+.
In contrast, some legacy configs require AI Layer to start in all safe modes, contradicting official docs.

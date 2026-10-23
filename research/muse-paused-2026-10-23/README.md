# Muse work paused (2026-10-23)

Paused at the user's request in favor of DeepSeek V4.1 Flash 1017.
Production implementation and prior validation are in `docs/MUSE_SPARK.md`;
last implementation commit is `56bc40d`. The 10 tokens/sec target remains unmet.
No model or benchmark process remains active. AC and battery power modes were
restored to Automatic (pmset lowpowermode 0 / highpowermode 0).

## Final experiments (not production)

The 100 W reported AC control measured 5.45 / 5.89 / 8.57 aggregate TPS for
Automatic / High / Automatic on the same verified saved 8K trajectory. All
55,159,104 FP32 values and 272 token IDs per run matched exactly; telemetry was
continuous. The change across Automatic controls prevents attributing a gain to
High Power. Same-code session allocation/order bias also needs future controls.

The archived Q6 shaders explore independent FP32 partial sums without narrowing
weights or inputs. All variants passed the unchanged 1e-4 absolute FP64 bound,
finite checks, offset guards, input nonmutation, and batch consistency for
1, 3, 33, and 512 rows, with 256 independent FP64 output samples per shape.
Initial two-chain quarter split measured +5.68% in the component; its replay
measured +0.19%. Four quarter chains were slower by 0.87%. Two half-block chains
measured +13.89%; alternating-block chains +0.99%. These are single component
runs, not established model throughput gains. No full-model drift or quality
validation was performed for these Q6 candidates; do not enable them in ds4.
The predeclared promotion gates are archived alongside the sources.

The probe sources include ds4_metal.m and link the normal non-Metal core objects;
compile with -O3 -fno-fast-math -fobjc-arc -mcpu=native -I. from the repository.
Pass MODEL, qkv.manifest, SHADER, and variant (2/4 for split, 2/6/8 for layouts).
They require the local 160-byte qkv.manifest (four 40-byte tensor records), not
included model data. The sampled Q6 record is offset 14295107936, 55050240 bytes,
type 14, input/output 8192. Existing native Metal runtime fast math is unchanged.

## Local evidence and resume

All raw artifacts, saved states, probes and measurement logs remain under:
`/Users/stephenbelanger/.ds4/validation/muse-2026-10-21`.
The model remains `/Users/stephenbelanger/.ds4/models/muse-spark-open-native.gguf`.
The final Q6 logs are `q6-split-{2,4}-green.*` and `q6-layout-{6,8,2}-green.*`.
Power runs are `power100-{auto-before,high,auto-after}-2026-10-23`.
The long clamshell-interrupted published-protocol run is invalid for performance.
Do not resume GPU tests from the older resume-performance.json automatically.
Keep the lid open and use benchmark_guard.py/run_display_only.py if resuming.

No private binary, weight, saved state, or raw reasoning trace is committed.
The preexisting untracked tests/test_ssd_cache is a compiled executable.

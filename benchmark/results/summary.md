# pyATS MCP transport benchmark

Connection setup time per condition:

| condition | connect_elapsed_s |
|---|---|
| pre_stdio (main, mcp1.26) | 1.0739 |
| post_modern_stateful (2.0, auto/SEP-2575) | 0.0245 |
| post_legacy_stateful (2.0, legacy proto) | 0.0227 |
| post_legacy_stateless (2.0, legacy proto, stateless_http) | 0.0257 |

## single_show_command

| condition | n | mean (ms) | p50 (ms) | p95 (ms) |
|---|---|---|---|---|
| pre_stdio (main, mcp1.26) | 8 | 15690.8 | 15678.0 | 15817.9 |
| post_modern_stateful (2.0, auto/SEP-2575) | 8 | 15697.2 | 15708.5 | 15818.1 |
| post_legacy_stateful (2.0, legacy proto) | 8 | 15690.9 | 15711.6 | 15775.8 |
| post_legacy_stateless (2.0, legacy proto, stateless_http) | 8 | 15678.1 | 15682.2 | 15752.1 |

## multi_show_thread_pool

| condition | n | mean (ms) | p50 (ms) | p95 (ms) |
|---|---|---|---|---|
| pre_stdio (main, mcp1.26) | 8 | 15765.7 | 15729.4 | 15894.9 |
| post_modern_stateful (2.0, auto/SEP-2575) | 8 | 15752.1 | 15754.4 | 15812.0 |
| post_legacy_stateful (2.0, legacy proto) | 8 | 15764.8 | 15716.7 | 15953.0 |
| post_legacy_stateless (2.0, legacy proto, stateless_http) | 8 | 15714.6 | 15695.4 | 15812.9 |

## multi_show_pcall

| condition | n | mean (ms) | p50 (ms) | p95 (ms) |
|---|---|---|---|---|
| pre_stdio (main, mcp1.26) | - | tool not available | - | - |
| post_modern_stateful (2.0, auto/SEP-2575) | 8 | 15884.4 | 15789.4 | 16045.5 |
| post_legacy_stateful (2.0, legacy proto) | 8 | 15852.5 | 15790.1 | 16040.4 |
| post_legacy_stateless (2.0, legacy proto, stateless_http) | 8 | 15883.7 | 15792.8 | 16040.7 |

## device_health

| condition | n | mean (ms) | p50 (ms) | p95 (ms) |
|---|---|---|---|---|
| pre_stdio (main, mcp1.26) | 8 | 18057.7 | 18054.4 | 18154.1 |
| post_modern_stateful (2.0, auto/SEP-2575) | 8 | 18131.5 | 18133.8 | 18243.1 |
| post_legacy_stateful (2.0, legacy proto) | 8 | 18179.5 | 18188.1 | 18327.8 |
| post_legacy_stateless (2.0, legacy proto, stateless_http) | 8 | 18111.6 | 18116.6 | 18231.7 |


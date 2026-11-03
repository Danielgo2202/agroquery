Run `ollama_qwen2.5-coder-3b` (qwen2.5-coder:3b / qwen2.5-coder:3b): 50 questions x 1 repeat(s), 200 evaluations, total cost $0.0000.

| Strategy | Accuracy (95% CI) | Easy | Medium | Hard | Avg input tokens | Avg output tokens | Avg cost / query | Cost / correct answer | Latency p50 | Escalated |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 Baseline | 44.0% (31%-58%) | 78% | 44% | 0% | 2,693 | 97 | $0.00000 | $0.00000 | 6.4 s | - |
| S2 Compact schema | 36.0% (24%-50%) | 67% | 33% | 0% | 1,260 | 152 | $0.00000 | $0.00000 | 11.0 s | - |
| S3 Self-correcting | 48.0% (35%-61%) | 83% | 50% | 0% | 4,704 | 186 | $0.00000 | $0.00000 | 7.6 s | - |
| S4 Model routing | 48.0% (35%-61%) | 83% | 50% | 0% | 7,294 | 304 | $0.00000 | $0.00000 | 7.6 s | 30% |

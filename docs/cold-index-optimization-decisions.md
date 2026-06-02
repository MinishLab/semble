# Cold index optimization decisions

本文记录 cold indexing 优化的入库判断。目标是避免把改变输出口径的改动误记为性能优化。

## Benchmark 口径

所有可计入性能优化的结论必须满足：

- benchmark target：本机私有大仓，由运行者在本地通过参数或环境变量指定；仓库内不记录具体路径
- cache delete target：Semble index cache directory，不删除 source repo、model cache 或 HF cache
- 不删除 model / HF cache
- full cold build：`SembleIndex.from_path(..., content=(ContentType.CODE,))`
- `BM25 + semantic` 都 ready 后才算完成
- chunk 数不变，才可称为 same-output perf win

如果 chunk 数、chunk granularity、tree-sitter fallback 语义发生变化，只能记录为 quality/cost tradeoff，不能计入 pure perf win。

## 已确认可入库：same-output perf win

### `92c28b0 Optimize chunking hot paths without changing granularity`

结论：可作为真实性能优化入库。

证据：focused chunk attribution benchmark。

| commit | label | chunks | elapsed |
|---|---|---:|---:|
| `e6afc1d` | baseline | `572873` | `1150.776s` |
| `92c28b0` | same-output-hot-paths | `572873` | `546.258s` |

判断：

- chunk 数相同：`572873 -> 572873`
- elapsed：`1150.776s -> 546.258s`
- delta：`-604.518s`
- 约 `-52.5%`

该 commit 不改变默认 chunk granularity。它优化的是 hot path：

- line-number 计算不再对每个 chunk 做 prefix scan
- worker thread 使用 thread-local tree-sitter parser，避免跨线程共享 parser
- token split 使用 bounded cache，避免 repeated identifier 重复 camel/snake split

入库理由：同输出口径下显著降低 cold build 时间。

### `255873e Add git source inventory planning`

结论：可作为 same-output full-cold 优化候选入库。

证据：isolated baseline + single-factor attribution。

| case | chunks | elapsed |
|---|---:|---:|
| `92c28b0` baseline | `572873` | `547.371s` |
| `92c28b0 + 255873e` | `572873` | `488.183s` |

判断：

- chunk 数相同：`572873 -> 572873`
- elapsed：`547.371s -> 488.183s`
- delta：`-59.188s`
- 约 `-10.8%`

入库理由：同输出口径下降低 full cold。该改动仍需代码审查确认收益来源，避免 source inventory module 里夹带其它 cold-path 行为。

### `f8e97cd Validate hybrid cache source metadata`

结论：benchmark 显示 same-output full-cold 有收益；但需代码审查确认是否夹带冷构建路径改动。

证据：isolated baseline + single-factor attribution。

| case | chunks | elapsed |
|---|---:|---:|
| `92c28b0` baseline | `572873` | `547.371s` |
| `92c28b0 + f8e97cd` | `572873` | `484.698s` |

判断：

- chunk 数相同：`572873 -> 572873`
- elapsed：`547.371s -> 484.698s`
- delta：`-62.673s`
- 约 `-11.4%`

注意：commit 名义是 cache/source metadata correctness；这类改动不应天然影响 delete-cache full cold。入库前必须读 diff，确认具体冷路径收益来自哪里。不能只凭名字归因。

## 不作为 pure perf 入库：chunk granularity tradeoff

### `6f7c45a Adjust chunk granularity for cold indexing`

结论：不能作为 pure perf win 入库。只能作为单独产品/质量-成本决策。

证据：focused chunk attribution benchmark。

| commit | label | chunks | elapsed |
|---|---|---:|---:|
| `92c28b0` | same-output-hot-paths | `572873` | `546.258s` |
| `6f7c45a` | chunk-granularity-tradeoff | `174901` | `297.099s` |

判断：

- chunk 数变化：`572873 -> 174901`
- 这是输出 granularity 变化，不是同口径性能提升
- elapsed 下降不能计入 pure perf attribution

该 commit 涉及：

- `_DESIRED_CHUNK_LENGTH_CHARS = 8000`
- `_TREE_SITTER_MAX_SOURCE_CHARS = 512_000`
- 大文件跳过 tree-sitter parser，走 line chunking

如果要入库，必须单独评估：

- retrieval granularity
- result location 精度
- recall / ranking 质量
- generated/noisy files 的成本收益

记录口径：可以说它降低 cold indexing 成本；不能说它是同输出性能优化。

## 不算大幅优化或单独看变慢

### `a25df5e Add index persistence primitives`

isolated single-factor：

| case | chunks | elapsed |
|---|---:|---:|
| `92c28b0` baseline | `572873` | `547.371s` |
| `92c28b0 + a25df5e` | `572873` | `542.008s` |

判断：`-5.363s`，幅度小，可能是噪声；不作为大幅 perf win。

第二轮 dependency-bound 复跑为 `536.120s`，仍只能说明小幅/噪声级改善，不作为主要优化。

### `3ca76dc Add LMDB chunk payload store`

isolated single-factor：

| case | chunks | elapsed |
|---|---:|---:|
| `92c28b0` baseline | `572873` | `547.371s` |
| `92c28b0 + 3ca76dc` | `572873` | `576.829s` |

判断：`+29.458s`，单独看 full cold 变慢。它是 persistence infrastructure，不是 cold wall-clock win。

## Dependency-bound attribution

这些因素不能用 `baseline + single factor` 完整回答，因为单独 patch 会缺依赖或不可运行。

### `c2cfa97 Add persistent sparse and dense backends`

单独跑失败：

```text
ModuleNotFoundError: No module named 'tantivy'
```

加 `a25df5e` 后仍失败：

```text
ModuleNotFoundError: No module named 'semble.index.chunk_store'
```

说明该因素至少依赖 persistence primitives 和 LMDB chunk store。不能声称单因素归因。

可运行 stack：

| case | commits | chunks | elapsed |
|---|---|---:|---:|
| baseline | `92c28b0` | `572873` | `547.371s` |
| pre-streaming stack | `a25df5e + 3ca76dc + 255873e + c2cfa97 + f8e97cd` | `572873` | `398.894s` |

判断：该 stack 在 same-output 口径下有明显收益：`-148.477s`。但这不是单因素结论，不能把收益全部归给 `c2cfa97`。

### `ae09e41 Integrate streaming hybrid index builds`

单独跑失败：

```text
ModuleNotFoundError: No module named 'lmdb'
```

dependency-bound 对比：

| case | commits | chunks | elapsed |
|---|---|---:|---:|
| pre-streaming stack | `a25df5e + 3ca76dc + 255873e + c2cfa97 + f8e97cd` | `572873` | `398.894s` |
| streaming stack | `a25df5e + 3ca76dc + 255873e + c2cfa97 + f8e97cd + ae09e41` | `558666` | `451.152s` |

判断：

- chunk 数变化：`572873 -> 558666`
- elapsed 变慢：`398.894s -> 451.152s`
- 因 chunk 数也变了，这不是 clean same-output comparison
- 即便忽略 chunk 数变化，streaming stack 在 full cold wall time 上也没有显示收益

记录口径：`ae09e41` 不能作为 full-cold wall-clock 优化入库；它的价值应按 memory pressure、persistence correctness、incremental rebuild、hybrid readiness 评估。

## 当前入库优先级

1. 入库：`92c28b0` — 明确大幅 same-output perf win。
2. 候选入库：`255873e` — same-output `-59.188s`，需代码审查确认范围。
3. 候选入库但需严审：`f8e97cd` — same-output `-62.673s`，名字与 cold-path 收益不直觉一致。
4. 不按 perf 入库：`6f7c45a` — chunk granularity tradeoff。
5. 不作为 cold wall-clock win：`3ca76dc`、`ae09e41`。
6. Stack-level 候选：`a25df5e + 3ca76dc + 255873e + c2cfa97 + f8e97cd` — same-output stack `-148.477s`，但需要进一步拆最小依赖再归因。

## 决策规则

1. Same chunk count + lower elapsed => 可称为性能优化。
2. Different chunk count + lower elapsed => 只能称为 granularity tradeoff。
3. Correctness / cache validity / source snapshot / hybrid readiness 改动不按 cold wall time 判断。
4. 每个待入库优化必须有独立 benchmark 证据，不能从 monolithic commit 推断。
5. Dependency-bound commit 只能做 stack attribution；不能偷换成 single-factor attribution。

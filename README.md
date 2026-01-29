# TUGraph

Kotlin + DJL playground for knowledge-graph embedding models and small DL demos.

## What’s inside

Core models (in `src/main/kotlin/jp/live/ugai/tugraph`):
- `TransE`, `TransR` (distance-based)
- `ComplEx`, `DistMult` (similarity-based)
- `EmbeddingTrainer` with negative sampling, Bernoulli head/tail corruption for TransE/TransR,
  k-negative sampling, and self-adversarial weighting for ComplEx/DistMult
- `ResultEval` for ranking metrics (HIT@K, MRR) with batched scoring

Examples / entry points:
- `Test6Kt` – TransE training + eval on `data/sample.csv` (default `application` main)
- `TestTransRKt`, `TestComplExKt`, `TestDistMultKt`
- `TestTransformerKt`, `EmbeddingExampleKt`, `BertExampleKt` (misc demos)

## Requirements

- Java 11
- Gradle (wrapper provided)

DJL is configured with PyTorch and a CUDA 12.4 native runtime in `build.gradle.kts`.

## Quick start

Build:
```
./gradlew compileKotlin
```

Run the default example (TransE, `Test6Kt`):
```
./gradlew run
```

Run a specific entry point:
```
./gradlew execute -PmainClass=jp.live.ugai.tugraph.TestComplExKt
```

## Data

Example CSV is in `data/sample.csv`. The CSV reader expects triples formatted as:
```
head,relation,tail
```

## Configuration

Common hyperparameters are in `src/main/kotlin/jp/live/ugai/tugraph/Commons.kt`:
- `DIMENSION`, `LEARNING_RATE`, `NEPOCH`, `BATCH_SIZE`
- `NEGATIVE_SAMPLES`, `SELF_ADVERSARIAL_TEMP`, `NEGATIVE_RESAMPLE_CAP`
- `RESULT_EVAL_BATCH_SIZE`
- `DISTMULT_L2`, `COMPLEX_L2`

## Linting / static analysis

```
./gradlew detekt
./gradlew ktlintCheck
```

Detekt config: `config/detekt.yml`

## Notes

- Distance-based models (TransE/TransR) use margin ranking loss.
- Similarity-based models (ComplEx/DistMult) use logistic loss with self-adversarial negative weighting.
- `ResultEval` supports both directions via `higherIsBetter` (set true for ComplEx/DistMult).

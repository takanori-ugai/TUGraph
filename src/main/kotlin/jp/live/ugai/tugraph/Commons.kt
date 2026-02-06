package jp.live.ugai.tugraph

import jp.live.ugai.tugraph.eval.*
// Hyper Parameters

/** Default learning rate used in training examples. */
const val LEARNING_RATE = 0.005f

/** Default Adagrad learning rate for embedding models. */
const val ADAGRAD_LEARNING_RATE = 0.1f

/** Default Adam learning rate for HyperComplEx. */
const val HYPERCOMPLEX_ADAM_LEARNING_RATE = 1.0e-3f

/** Adam beta1 for HyperComplEx. */
const val HYPERCOMPLEX_ADAM_BETA1 = 0.9f

/** Adam beta2 for HyperComplEx. */
const val HYPERCOMPLEX_ADAM_BETA2 = 0.999f

/** Adam epsilon for HyperComplEx. */
const val HYPERCOMPLEX_ADAM_EPSILON = 1.0e-8f

/** Gradient clipping threshold for HyperComplEx Adam optimizer (0 disables). */
const val HYPERCOMPLEX_ADAM_CLIP_GRAD = 1.0f

/** Default embedding dimension for example models. */
const val DIMENSION = 200L

/** Default number of epochs for example training runs. */
const val NEPOCH = 30

/** Default batch size for example datasets. */
const val BATCH_SIZE = 1000

/** Max attempts to resample a negative before falling back. */
const val NEGATIVE_RESAMPLE_CAP = 10

/** Batch size for ResultEval scoring. */
const val RESULT_EVAL_BATCH_SIZE = 8

/** Chunk size for entity candidates during ResultEval scoring to limit memory use. */
const val RESULT_EVAL_ENTITY_CHUNK_SIZE = 8000

/** Run validation every N epochs during training. */
const val EVAL_EVERY = 5

/** L2 regularization weight for DistMult embeddings. */
const val DISTMULT_L2 = 1.0e-4f

/** L2 regularization weight for ComplEx embeddings. */
const val COMPLEX_L2 = 1.0e-4f

/** Hyperbolic curvature for HyperComplEx (Poincare ball). */
const val HYPERCOMPLEX_CURVATURE = 1.0f

/** Margin gamma for HyperComplEx self-adversarial ranking loss. */
const val HYPERCOMPLEX_GAMMA = 12.0f

/** Temperature beta for HyperComplEx self-adversarial ranking loss. */
const val HYPERCOMPLEX_BETA = 1.0f

/** Consistency loss weight for HyperComplEx. */
const val HYPERCOMPLEX_LAMBDA1 = 0.1f

/** L2 regularization weight for HyperComplEx. */
const val HYPERCOMPLEX_LAMBDA2 = 1.0e-4f

/** Default number of entity shards for HyperComplEx. */
const val HYPERCOMPLEX_ENTITY_SHARDS = 4

/** Default number of relation shards for HyperComplEx. */
const val HYPERCOMPLEX_RELATION_SHARDS = 1

/** Enable mixed-precision training for HyperComplEx. */
const val HYPERCOMPLEX_MIXED_PRECISION = true

/** Loss scale for mixed-precision HyperComplEx training. */
const val HYPERCOMPLEX_LOSS_SCALE = 1024.0f

/** Adaptive dimension allocation result for HyperComplEx. */
data class HyperComplExDims(val dH: Long, val dC: Long, val dE: Long)

/**
 * Allocate HyperComplEx dimensions based on graph size while keeping a constant base budget.
 *
 * The paper specifies dH, dC, dE = f(|E|, Dbase) but does not define f explicitly.
 * This implementation uses a smooth interpolation over log10(|E|) that shifts capacity
 * toward hyperbolic dimensions for larger graphs while maintaining dH + dC + dE = Dbase.
 */
fun adaptiveHyperComplExDims(
    numEntities: Long,
    baseDim: Long,
): HyperComplExDims {
    require(baseDim >= 3) { "baseDim must be >= 3 to allocate across spaces (baseDim=$baseDim)." }
    val logE = kotlin.math.log10(kotlin.math.max(1.0, numEntities.toDouble()))
    val t = ((logE - 3.0) / 5.0).coerceIn(0.0, 1.0).toFloat() // 1e3 -> 0, 1e8 -> 1
    val min = floatArrayOf(0.2f, 0.4f, 0.4f) // small graphs
    val max = floatArrayOf(0.5f, 0.3f, 0.2f) // large graphs
    val ratios =
        FloatArray(3) { i ->
            (min[i] + (max[i] - min[i]) * t).coerceAtLeast(0.01f)
        }
    val sum = ratios.sum()
    for (i in ratios.indices) ratios[i] /= sum

    var dH = kotlin.math.round(baseDim * ratios[0]).toLong()
    var dC = kotlin.math.round(baseDim * ratios[1]).toLong()
    var dE = kotlin.math.round(baseDim * ratios[2]).toLong()
    if (dH < 1L) dH = 1
    if (dC < 1L) dC = 1
    if (dE < 1L) dE = 1

    var total = dH + dC + dE
    if (total != baseDim) {
        val diff = baseDim - total
        if (diff > 0) {
            when {
                dH <= dC && dH <= dE -> dH += diff
                dC <= dH && dC <= dE -> dC += diff
                else -> dE += diff
            }
        } else {
            var remaining = -diff
            while (remaining > 0) {
                when {
                    dH >= dC && dH >= dE && dH > 1 -> dH--
                    dC >= dH && dC >= dE && dC > 1 -> dC--
                    dE > 1 -> dE--
                    else -> break
                }
                remaining--
            }
        }
    }
    return HyperComplExDims(dH, dC, dE)
}

/** Number of negative samples per positive triple. */
const val NEGATIVE_SAMPLES = 1

/** Self-adversarial temperature for similarity-based models. */
const val SELF_ADVERSARIAL_TEMP = 0.0f

/** Number of values in a knowledge graph triple. */
const val TRIPLE = 3L

/** Matryoshka dimensions for RotatE (total embedding size). */
val MATRYOSHKA_ROTATE_DIMS = longArrayOf(DIMENSION, DIMENSION * 2)

/** Matryoshka weights for RotatE (must align with MATRYOSHKA_ROTATE_DIMS). */
val MATRYOSHKA_ROTATE_WEIGHTS = floatArrayOf(0.3f, 0.7f)

/** Matryoshka dimensions for QuatE (total embedding size). */
val MATRYOSHKA_QUATE_DIMS = longArrayOf(DIMENSION, DIMENSION * 2, DIMENSION * 4)

/** Matryoshka weights for QuatE (must align with MATRYOSHKA_QUATE_DIMS). */
val MATRYOSHKA_QUATE_WEIGHTS = floatArrayOf(0.2f, 0.3f, 0.5f)

// const val NUM_ENTITIES = 202487L
// const val NUM_ENTITIES = 7658L
// const val NUM_EDGES = 37L
// const val NUM_EDGES = 57L

/** Placeholder class to keep common constants in one file. */
class Commons

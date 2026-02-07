package jp.live.ugai.tugraph.train

/**
 * Immutable configuration derived from the model and global settings for a training run.
 *
 * @property batchSize Triples per batch.
 * @property evalEvery Evaluation interval (epochs).
 * @property numNegatives Negative samples per positive triple.
 * @property numTriplesInt Total number of triples as an Int.
 * @property adapter Model adapter used for post-update hooks.
 * @property useSimilarityLoss Whether to use similarity-based loss instead of hinge.
 * @property higherIsBetter Whether higher scores indicate better predictions.
 * @property useSelfAdversarial Whether to weight negatives with self-adversarial sampling.
 * @property useMatryoshka Whether to use Matryoshka loss (supported blocks only).
 * @property useBernoulli Whether to use Bernoulli negative sampling.
 * @property regWeight L2 regularization weight.
 */
internal data class TrainingPlan(
    val batchSize: Int,
    val evalEvery: Int,
    val numNegatives: Int,
    val numTriplesInt: Int,
    val adapter: ModelAdapter,
    val useSimilarityLoss: Boolean,
    val higherIsBetter: Boolean,
    val useSelfAdversarial: Boolean,
    val useMatryoshka: Boolean,
    val useBernoulli: Boolean,
    val regWeight: Float,
)

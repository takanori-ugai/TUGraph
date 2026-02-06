package jp.live.ugai.tugraph.train

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

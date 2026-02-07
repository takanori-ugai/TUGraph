package jp.live.ugai.tugraph.train

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.training.Trainer
import jp.live.ugai.tugraph.BATCH_SIZE
import jp.live.ugai.tugraph.COMPLEX_L2
import jp.live.ugai.tugraph.ComplEx
import jp.live.ugai.tugraph.DISTMULT_L2
import jp.live.ugai.tugraph.DistMult
import jp.live.ugai.tugraph.EVAL_EVERY
import jp.live.ugai.tugraph.HyperComplEx
import jp.live.ugai.tugraph.NEGATIVE_SAMPLES
import jp.live.ugai.tugraph.QuatE
import jp.live.ugai.tugraph.RotatE
import jp.live.ugai.tugraph.SELF_ADVERSARIAL_TEMP
import jp.live.ugai.tugraph.TRIPLE
import jp.live.ugai.tugraph.TransE
import jp.live.ugai.tugraph.TransR
import kotlin.math.min

/**
 * Orchestrates training for a knowledge-graph embedding model.
 *
 * @param manager NDManager used to allocate and own training arrays.
 * @param rawTriples NDArray containing training triples (shape [numTriples, 3] or flat multiple of 3).
 * @param numOfEntities Number of entities in the dataset.
 * @param trainer DJL trainer configured with the model and optimizer.
 * @param epoch Number of epochs to train.
 * @param useMatryoshkaOverride Enables Matryoshka loss for supported blocks when true.
 */
class EmbeddingTrainer(
    private val manager: NDManager,
    rawTriples: NDArray,
    private val numOfEntities: Long,
    private val trainer: Trainer,
    private val epoch: Int,
    private val useMatryoshkaOverride: Boolean = false,
) : java.io.Closeable {
    private val triples: NDArray

    private val logger = org.slf4j.LoggerFactory.getLogger(EmbeddingTrainer::class.java)

    /** Per-epoch loss values collected during training. */
    private val _lossList = mutableListOf<Float>()
    val lossList: List<Float> get() = _lossList
    private val numOfTriples: Long

    /** Materialized triples used for negative sampling. */
    private val _inputList = mutableListOf<LongArray>()
    val inputList: List<LongArray> get() = _inputList
    private val inputSet: Set<TripleKey>

    private val bernoulliProb: Map<Long, Float>
    private val sampler: NegativeSampler
    private val losses = EmbeddingLosses(trainer)

    init {
        val tripleCount = rawTriples.size() / TRIPLE
        require(tripleCount * TRIPLE == rawTriples.size()) {
            "Triples NDArray size must be a multiple of TRIPLE; size=${rawTriples.size()}."
        }
        numOfTriples = tripleCount
        triples =
            if (rawTriples.shape.dimension() == 1 || rawTriples.shape[1] != TRIPLE) {
                rawTriples.reshape(tripleCount, TRIPLE).also { it.attach(manager) }
            } else {
                rawTriples
            }
        require(numOfTriples <= Int.MAX_VALUE.toLong()) {
            "Dataset exceeds Int.MAX_VALUE triples; Long indexing required (numOfTriples=$numOfTriples)."
        }
        val totalTriples = numOfTriples.toInt()
        val flat = triples.toLongArray()
        val relCounts = mutableMapOf<Long, Int>()
        val headSets = mutableMapOf<Long, MutableSet<Long>>()
        val tailSets = mutableMapOf<Long, MutableSet<Long>>()
        val inputSetMutable = HashSet<TripleKey>(totalTriples * 2)

        var idx = 0
        repeat(totalTriples) {
            val head = flat[idx]
            val rel = flat[idx + 1]
            val tail = flat[idx + 2]
            idx += 3
            val triple = longArrayOf(head, rel, tail)
            _inputList.add(triple)
            inputSetMutable.add(TripleKey(head, rel, tail))
            relCounts[rel] = (relCounts[rel] ?: 0) + 1
            headSets.getOrPut(rel) { mutableSetOf() }.add(head)
            tailSets.getOrPut(rel) { mutableSetOf() }.add(tail)
        }
        inputSet = inputSetMutable
        bernoulliProb =
            relCounts.mapValues { (rel, count) ->
                val heads = headSets[rel]?.size ?: 1
                val tails = tailSets[rel]?.size ?: 1
                val tph = count.toFloat() / heads
                val hpt = count.toFloat() / tails
                tph / (tph + hpt)
            }
        sampler = NegativeSampler(numOfEntities, bernoulliProb, inputSet, logger)
    }

    /**
     * Trains the embedding model for the configured number of epochs.
     *
     * The loop batches triples, generates negatives, computes the configured loss, and updates parameters.
     * It also tracks metrics, applies post-update normalization/projection when needed, and logs anomalies.
     */
    fun training() {
        validateInputs()
        val plan = buildTrainingPlan()
        val state = TrainingState()
        val logEvery = maxOf(1, epoch / 500L)
        for (epochIndex in 1..epoch) {
            val epochLoss = trainEpoch(plan, state)
            val trainAvg = epochLoss / numOfTriples
            if (epochIndex % plan.evalEvery == 0 || epochIndex == epoch) {
                state.lastValidate =
                    evaluateEpochLoss(triples, plan)
            }
            maybeLogEpoch(epochIndex, trainAvg, logEvery)
            _lossList.add(epochLoss)
            reportMetrics(trainAvg, state.lastValidate)
            trainer.notifyListeners { listener -> listener.onEpoch(trainer) }
        }
    }

    private fun validateInputs() {
        val numTriplesInt = numOfTriples.toInt()
        require(numTriplesInt > 0) { "No triples available for training (numOfTriples=0)." }
    }

    private fun buildTrainingPlan(): TrainingPlan {
        val batchSize = maxOf(1, BATCH_SIZE)
        val evalEvery = maxOf(1, EVAL_EVERY)
        val numNegatives = maxOf(1, NEGATIVE_SAMPLES)
        val numTriplesInt = numOfTriples.toInt()
        val adapter = ModelAdapter.from(trainer.model.block)
        val block = adapter.block
        val useSimilarityLoss =
            when (block) {
                is ComplEx, is DistMult -> true
                else -> false
            }
        val higherIsBetter =
            when (block) {
                is ComplEx, is DistMult, is QuatE, is HyperComplEx -> true
                else -> false
            }
        val useSelfAdversarial =
            when (block) {
                is ComplEx, is DistMult -> true
                else -> false
            } &&
                SELF_ADVERSARIAL_TEMP > 0.0f
        val useMatryoshkaSupported =
            when (block) {
                is RotatE, is QuatE -> true
                else -> false
            }
        val useMatryoshka =
            if (useMatryoshkaOverride) {
                if (!useMatryoshkaSupported && logger.isWarnEnabled) {
                    logger.warn("Matryoshka override enabled for unsupported block {}.", block::class.simpleName)
                }
                useMatryoshkaSupported
            } else {
                false
            }
        val useBernoulli =
            when (block) {
                is TransE, is TransR -> true
                else -> false
            }
        val regWeight =
            when (block) {
                is ComplEx -> COMPLEX_L2
                is DistMult -> DISTMULT_L2
                else -> 0.0f
            }
        return TrainingPlan(
            batchSize = batchSize,
            evalEvery = evalEvery,
            numNegatives = numNegatives,
            numTriplesInt = numTriplesInt,
            adapter = adapter,
            useSimilarityLoss = useSimilarityLoss,
            higherIsBetter = higherIsBetter,
            useSelfAdversarial = useSelfAdversarial,
            useMatryoshka = useMatryoshka,
            useBernoulli = useBernoulli,
            regWeight = regWeight,
        )
    }

    private fun trainEpoch(
        plan: TrainingPlan,
        state: TrainingState,
    ): Float {
        var epochLoss = 0f
        var start = 0
        while (start < plan.numTriplesInt) {
            val end = min(start + plan.batchSize, plan.numTriplesInt)
            epochLoss += trainBatch(plan, state, start, end)
            stepAndPostUpdate(plan, start, end)
            start = end
        }
        return epochLoss
    }

    private fun trainBatch(
        plan: TrainingPlan,
        state: TrainingState,
        start: Int,
        end: Int,
    ): Float {
        var batchLossSum = 0f
        manager.newSubManager().use { batchManager ->
            val sample =
                triples.get(NDIndex("$start:$end")).also {
                    it.attach(batchManager)
                }
            val negativeSample =
                sampler.sample(
                    batchManager,
                    sample,
                    plan.numNegatives,
                    plan.useBernoulli,
                )
            trainer.newGradientCollector().use { gc ->
                if (plan.adapter is HyperComplExAdapter) {
                    val hyperBlock = plan.adapter.block
                    val lossResult =
                        losses.computeHyperComplExLossWithScores(
                            hyperBlock,
                            sample,
                            negativeSample,
                            plan.numNegatives,
                            training = true,
                        )
                    try {
                        lossResult.loss.use { lossValue ->
                            if (!state.firstNanReported && logger.isWarnEnabled) {
                                lossResult.posScores.isNaN().any().use { posNaNArr ->
                                    lossResult.negScores.isNaN().any().use { negNaNArr ->
                                        lossValue.isNaN().any().use { lossNaNArr ->
                                            val posNaN = posNaNArr.getBoolean()
                                            val negNaN = negNaNArr.getBoolean()
                                            val lossNaN = lossNaNArr.getBoolean()
                                            if (posNaN || negNaN || lossNaN) {
                                                logger.warn(
                                                    "NaN detected in batch {}:{} (posScores={}, negScores={}, loss={})",
                                                    start,
                                                    end,
                                                    posNaN,
                                                    negNaN,
                                                    lossNaN,
                                                )
                                                logger.warn("Sample: {}", sample)
                                                logger.warn("Negative sample: {}", negativeSample)
                                                state.firstNanReported = true
                                            }
                                        }
                                    }
                                }
                            }
                            lossValue.mean().use { meanLoss ->
                                val batchLoss = meanLoss.getFloat()
                                lossValue.sum().use { lossSum ->
                                    batchLossSum += lossSum.getFloat()
                                }
                                trainer.metrics?.addMetric("train_loss", batchLoss)
                                trainer.metrics?.addMetric("train_L1Loss", batchLoss)
                                val lossForBackward =
                                    if (hyperBlock.mixedPrecision) {
                                        meanLoss.mul(hyperBlock.lossScale)
                                    } else {
                                        meanLoss
                                    }
                                try {
                                    gc.backward(lossForBackward)
                                } finally {
                                    if (lossForBackward !== meanLoss) {
                                        lossForBackward.close()
                                    }
                                }
                                if (hyperBlock.mixedPrecision) {
                                    hyperBlock.unscaleGradients(hyperBlock.lossScale)
                                }
                            }
                        }
                    } finally {
                        lossResult.posScores.close()
                        lossResult.negScores.close()
                    }
                } else {
                    trainer.forward(NDList(sample, negativeSample)).use { f0 ->
                        f0.forEach { it.attach(batchManager) }
                        val pos = f0[0]
                        val neg =
                            f0[1].reshape(pos.shape[0], plan.numNegatives.toLong()).also {
                                it.attach(batchManager)
                            }
                        var lossValue =
                            if (plan.useMatryoshka) {
                                losses.computeMatryoshkaLoss(
                                    plan.adapter.block,
                                    sample,
                                    negativeSample,
                                    plan.numNegatives,
                                    plan.useSimilarityLoss,
                                    plan.useSelfAdversarial,
                                    plan.higherIsBetter,
                                )
                            } else {
                                losses.computeLossFromScores(
                                    pos,
                                    neg,
                                    plan.useSimilarityLoss,
                                    plan.useSelfAdversarial,
                                    plan.higherIsBetter,
                                )
                            }
                        var baseLoss: NDArray? = null
                        if (plan.regWeight > 0.0f) {
                            val reg = computeL2Reg(plan.regWeight)
                            val newLoss = lossValue.add(reg)
                            reg.close()
                            baseLoss = lossValue
                            lossValue = newLoss
                        }
                        if (!state.firstNanReported && logger.isWarnEnabled) {
                            pos.isNaN().any().use { posNaNArr ->
                                neg.isNaN().any().use { negNaNArr ->
                                    lossValue.isNaN().any().use { lossNaNArr ->
                                        val posNaN = posNaNArr.getBoolean()
                                        val negNaN = negNaNArr.getBoolean()
                                        val lossNaN = lossNaNArr.getBoolean()
                                        if (posNaN || negNaN || lossNaN) {
                                            logger.warn(
                                                "NaN detected in batch {}:{} (pos={}, neg={}, loss={})",
                                                start,
                                                end,
                                                posNaN,
                                                negNaN,
                                                lossNaN,
                                            )
                                            logger.warn("Sample: {}", sample)
                                            logger.warn("Negative sample: {}", negativeSample)
                                            state.firstNanReported = true
                                        }
                                    }
                                }
                            }
                        }
                        try {
                            lossValue.mean().use { meanLoss ->
                                val batchLoss = meanLoss.getFloat()
                                lossValue.sum().use { lossSum ->
                                    batchLossSum += lossSum.getFloat()
                                }
                                trainer.metrics?.addMetric("train_loss", batchLoss)
                                trainer.metrics?.addMetric("train_L1Loss", batchLoss)
                                gc.backward(meanLoss)
                            }
                        } finally {
                            lossValue.close()
                            baseLoss?.close()
                        }
                    }
                }
            }
        }
        return batchLossSum
    }

    private fun stepAndPostUpdate(
        plan: TrainingPlan,
        start: Int,
        end: Int,
    ) {
        val stepCompleted =
            try {
                trainer.step()
                true
            } catch (e: IllegalStateException) {
                // DJL throws an IllegalStateException with this message when gradients are all zeros (observed in 0.26.x).
                if (e.message?.contains("Gradient values are all zeros") == true) {
                    logger.debug("Skipping trainer.step() due to zero gradients in batch {}:{}", start, end)
                    false
                } else {
                    throw e
                }
            }
        if (stepCompleted) {
            plan.adapter.postUpdateHook()
        }
    }

    private fun maybeLogEpoch(
        epochIndex: Int,
        trainAvg: Float,
        logEvery: Long,
    ) {
        if (epochIndex % logEvery == 0L) {
            logger.debug("Epoch {} finished. (L2Loss: {})", epochIndex, trainAvg)
        }
    }

    private fun reportMetrics(
        trainAvg: Float,
        lastValidate: Float,
    ) {
        if (!lastValidate.isNaN()) {
            trainer.metrics?.addMetric("validate_loss", lastValidate)
            trainer.metrics?.addMetric("validate_L1Loss", lastValidate)
        }
        trainer.notifyListeners { listener ->
            if (listener is HingeLossLoggingListener) {
                listener.updateEvaluations(trainAvg, lastValidate)
            }
        }
    }

    /**
     * Releases resources held by this trainer by closing the underlying NDManager.
     *
     * After calling this method, NDArrays attached to this manager are invalid and must not be used.
     */
    override fun close() {
        manager.close()
    }

    /**
     * Computes the average loss over the provided triples using the current model and plan settings.
     *
     * Uses the same negative sampling and loss configuration as training, without gradient updates.
     *
     * @param data NDArray of triples to evaluate (shape: [totalTriples, 3]).
     * @param plan Training settings used to evaluate the loss.
     * @return The average loss per triple across `data`.
     */
    private fun evaluateEpochLoss(
        data: NDArray,
        plan: TrainingPlan,
    ): Float {
        var lossSum = 0f
        var start = 0
        while (start < plan.numTriplesInt) {
            val end = min(start + plan.batchSize, plan.numTriplesInt)
            manager.newSubManager().use { batchManager ->
                val sample =
                    data.get(NDIndex("$start:$end")).also {
                        it.attach(batchManager)
                    }
                val negativeSample =
                    sampler.sample(
                        batchManager,
                        sample,
                        plan.numNegatives,
                        plan.useBernoulli,
                    )
                if (plan.adapter is HyperComplExAdapter) {
                    val lossValue =
                        losses.computeHyperComplExLoss(
                            plan.adapter.block,
                            sample,
                            negativeSample,
                            plan.numNegatives,
                            training = false,
                        )
                    lossValue.use { lv ->
                        lv.sum().use { lossSumNd ->
                            lossSum += lossSumNd.getFloat()
                        }
                    }
                } else {
                    trainer.evaluate(NDList(sample, negativeSample)).use { f0 ->
                        f0.forEach { it.attach(batchManager) }
                        val pos = f0[0]
                        val neg =
                            f0[1].reshape(pos.shape[0], plan.numNegatives.toLong()).also {
                                it.attach(batchManager)
                            }
                        var lossValue =
                            if (plan.useMatryoshka) {
                                losses.computeMatryoshkaLoss(
                                    plan.adapter.block,
                                    sample,
                                    negativeSample,
                                    plan.numNegatives,
                                    plan.useSimilarityLoss,
                                    plan.useSelfAdversarial,
                                    plan.higherIsBetter,
                                )
                            } else {
                                losses.computeLossFromScores(
                                    pos,
                                    neg,
                                    plan.useSimilarityLoss,
                                    plan.useSelfAdversarial,
                                    plan.higherIsBetter,
                                )
                            }
                        var baseLoss: NDArray? = null
                        if (plan.regWeight > 0.0f) {
                            val reg = computeL2Reg(plan.regWeight)
                            val newLoss = lossValue.add(reg)
                            reg.close()
                            baseLoss = lossValue
                            lossValue = newLoss
                        }
                        try {
                            lossValue.sum().use { lossSumNd ->
                                lossSum += lossSumNd.getFloat()
                            }
                        } finally {
                            lossValue.close()
                            baseLoss?.close()
                        }
                    }
                }
            }
            start = end
        }
        return lossSum / plan.numTriplesInt
    }

    private fun computeL2Reg(regWeight: Float): NDArray {
        val block = trainer.model.block
        val pow0 =
            block.parameters
                .valueAt(0)
                .array
                .pow(2)
        val pow1 =
            block.parameters
                .valueAt(1)
                .array
                .pow(2)
        return try {
            pow0.sum().use { reg1 ->
                pow1.sum().use { reg2 ->
                    reg1.add(reg2).use { sum ->
                        sum.mul(regWeight)
                    }
                }
            }
        } finally {
            pow0.close()
            pow1.close()
        }
    }
}

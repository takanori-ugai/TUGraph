package jp.live.ugai.tugraph

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.Shape
import ai.djl.training.Trainer
import kotlin.math.min

/**
 * A class for training an embedding model.
 *
 * @property manager The NDManager instance used for creating NDArrays.
 * @property triples The NDArray containing the training triples.
 * @property numOfEntities The number of entities in the training data.
 * @property trainer The Trainer instance for training the model.
 * @property epoch The number of training epochs.
 */
class EmbeddingTrainer(
    private val manager: NDManager,
    private val triples: NDArray,
    private val numOfEntities: Long,
    private val trainer: Trainer,
    private val epoch: Int,
) {
    private data class MatryoshkaConfig(
        val dims: LongArray,
        val weights: FloatArray,
        val entities: NDArray,
        val edges: NDArray,
    )

    private val margin = 1.0f
    private val logger = org.slf4j.LoggerFactory.getLogger(EmbeddingTrainer::class.java)

    /** Per-epoch loss values collected during training. */
    val lossList = mutableListOf<Float>()
    private val numOfTriples = triples.shape[0]

    /** Materialized triples used for negative sampling. */
    val inputList = mutableListOf<LongArray>()
    private val inputSet: Set<TripleKey>

    private data class TripleKey(val head: Long, val rel: Long, val tail: Long)

    private val bernoulliProb: Map<Long, Float>

    init {
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
            inputList.add(triple)
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
    }

    /**
     * Train the embedding model for the configured number of epochs using the provided triples and trainer.
     *
     * Performs epoch-based, batch training with vectorized negative sampling; computes either a
     * similarity-style or hinge-style loss (optionally using self-adversarial weighting); applies
     * L2 regularization when enabled; updates trainer metrics and notifies listeners each epoch;
     * normalizes translation-style model blocks after parameter updates; and closes the trainer when finished.
     */
    fun training() {
        val batchSize = maxOf(1, BATCH_SIZE)
        val evalEvery = maxOf(1, EVAL_EVERY)
        require(numOfTriples <= Int.MAX_VALUE.toLong()) {
            "Dataset exceeds Int.MAX_VALUE triples; Long indexing required (numOfTriples=$numOfTriples)."
        }
        val numTriplesInt = numOfTriples.toInt()
        require(numTriplesInt > 0) { "No triples available for training (numOfTriples=0)." }
        var firstNanReported = false
        val block = trainer.model.block
        val useSimilarityLoss =
            when (block) {
                is ComplEx, is DistMult -> true
                else -> false
            }
        val useSelfAdversarial =
            when (block) {
                is ComplEx, is DistMult -> true
                else -> false
            } &&
                SELF_ADVERSARIAL_TEMP > 0.0f
        val useMatryoshka =
            when (block) {
                is RotatE, is QuatE -> true
                else -> false
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
        val numNegatives = maxOf(1, NEGATIVE_SAMPLES)
        var lastValidate = Float.NaN
        for (epochIndex in 1..epoch) {
            var epochLoss = 0f
            var start = 0
            while (start < numTriplesInt) {
                val end = min(start + batchSize, numTriplesInt)
                manager.newSubManager().use { batchManager ->
                    val sample =
                        triples.get(NDIndex("$start:$end")).also {
                            it.attach(batchManager)
                        }
                    val negativeSample =
                        sampleNegatives(
                            batchManager,
                            sample,
                            numOfEntities,
                            numNegatives,
                            useBernoulli,
                        )
                    trainer.newGradientCollector().use { gc ->
                        trainer.forward(NDList(sample, negativeSample)).use { f0 ->
                            f0.forEach { it.attach(batchManager) }
                            val pos = f0[0]
                            val neg = f0[1].reshape(pos.shape[0], numNegatives.toLong())
                            var lossValue =
                                if (useMatryoshka) {
                                    computeMatryoshkaLoss(
                                        block,
                                        sample,
                                        negativeSample,
                                        numNegatives,
                                        useSimilarityLoss,
                                        useSelfAdversarial,
                                    )
                                } else {
                                    computeLossFromScores(pos, neg, useSimilarityLoss, useSelfAdversarial)
                                }
                            if (regWeight > 0.0f) {
                                block.parameters.valueAt(0).array.pow(2).use { pow1 ->
                                    pow1.sum().use { reg1 ->
                                        block.parameters.valueAt(1).array.pow(2).use { pow2 ->
                                            pow2.sum().use { reg2 ->
                                                reg1.add(reg2).use { regSum ->
                                                    regSum.mul(regWeight).use { reg ->
                                                        lossValue = lossValue.add(reg)
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            if (!firstNanReported && logger.isWarnEnabled) {
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
                                                firstNanReported = true
                                            }
                                        }
                                    }
                                }
                            }
                            lossValue.mean().use { meanLoss ->
                                val batchLoss = meanLoss.getFloat()
                                lossValue.sum().use { lossSum ->
                                    epochLoss += lossSum.getFloat()
                                }
                                trainer.metrics?.addMetric("train_loss", batchLoss)
                                trainer.metrics?.addMetric("train_L1Loss", batchLoss)
                                gc.backward(meanLoss)
                            }
                        }
                    }
                }
                trainer.step()
                when (val block = trainer.model.block) {
                    is TransE -> block.normalize()
                    is TransR -> block.normalize()
                }
                start = end
            }
            val trainAvg = epochLoss / numOfTriples
            if (epochIndex % evalEvery == 0 || epochIndex == epoch) {
                lastValidate =
                    evaluateEpochLoss(
                        triples,
                        numTriplesInt,
                        batchSize,
                        numOfEntities,
                        numNegatives,
                        useBernoulli,
                        useSimilarityLoss,
                        useSelfAdversarial,
                        regWeight,
                    )
            }
            val logEvery = maxOf(1, epoch / 500L)
            if (epochIndex % logEvery == 0L) {
                logger.debug("Epoch {} finished. (L2Loss: {})", epochIndex, trainAvg)
            }
            lossList.add(epochLoss)
            if (!lastValidate.isNaN()) {
                trainer.metrics?.addMetric("validate_loss", lastValidate)
                trainer.metrics?.addMetric("validate_L1Loss", lastValidate)
            }
            trainer.notifyListeners { listener ->
                if (listener is HingeLossLoggingListener) {
                    listener.updateEvaluations(trainAvg, lastValidate)
                }
            }
            trainer.notifyListeners { listener -> listener.onEpoch(trainer) }
        }
        trainer.close()
    }

    /**
     * Releases resources held by this trainer by closing the underlying NDManager.
     *
     * After calling this method the trainer's manager and any NDArrays attached to it become invalid and
     * should not be used.
     */
    fun close() {
        manager.close()
    }

    /**
     * Computes the average training loss over the given triples using the current model and negative sampling settings.
     *
     * Evaluates loss in batches, generating `numNegatives` negatives per triple with optional Bernoulli sampling,
     * and using either similarity-based loss (with optional self-adversarial weighting) or hinge loss. Applies
     * L2 regularization to model block parameters when `regWeight` > 0.
     *
     * @param data NDArray of triples to evaluate (shape: [totalTriples, 3]).
     * @param totalTriples Total number of triples contained in `data`.
     * @param batchSize Number of triples processed per evaluation batch.
     * @param numEntities Total number of entities available for negative sampling.
     * @param numNegatives Number of negative samples generated per positive triple.
     * @param useBernoulli If true, use relation-specific Bernoulli probabilities when deciding whether to corrupt
     *        head or tail.
     * @param useSimilarityLoss If true, compute similarity-based loss; otherwise compute hinge (margin) loss.
     * @param useSelfAdversarial If true and using similarity loss, weight negatives by self-adversarial softmax.
     * @param regWeight L2 regularization weight applied to the model block parameters (0 disables regularization).
     * @return The average loss per triple across `data`.
     */
    private fun evaluateEpochLoss(
        data: NDArray,
        totalTriples: Int,
        batchSize: Int,
        numEntities: Long,
        numNegatives: Int,
        useBernoulli: Boolean,
        useSimilarityLoss: Boolean,
        useSelfAdversarial: Boolean,
        regWeight: Float,
    ): Float {
        var lossSum = 0f
        var start = 0
        while (start < totalTriples) {
            val end = min(start + batchSize, totalTriples)
            manager.newSubManager().use { batchManager ->
                val sample =
                    data.get(NDIndex("$start:$end")).also {
                        it.attach(batchManager)
                    }
                val negativeSample =
                    sampleNegatives(
                        batchManager,
                        sample,
                        numEntities,
                        numNegatives,
                        useBernoulli,
                    )
                trainer.evaluate(NDList(sample, negativeSample)).use { f0 ->
                    f0.forEach { it.attach(batchManager) }
                    val pos = f0[0]
                    val neg = f0[1].reshape(pos.shape[0], numNegatives.toLong())
                    var lossValue =
                        if (useSimilarityLoss) {
                            val posLoss = pos.neg().exp().add(1f).log()
                            val negLoss = neg.exp().add(1f).log()
                            val negAgg =
                                if (useSelfAdversarial) {
                                    val weights = neg.mul(SELF_ADVERSARIAL_TEMP).softmax(1)
                                    weights.mul(negLoss).sum(intArrayOf(1))
                                } else {
                                    negLoss.mean(intArrayOf(1))
                                }
                            posLoss.add(negAgg)
                        } else {
                            val hinge = pos.expandDims(1).sub(neg).add(margin).maximum(0f)
                            hinge.mean(intArrayOf(1))
                        }
                    if (regWeight > 0.0f) {
                        trainer.model.block.parameters.valueAt(0).array.pow(2).use { pow1 ->
                            pow1.sum().use { reg1 ->
                                trainer.model.block.parameters.valueAt(1).array.pow(2).use { pow2 ->
                                    pow2.sum().use { reg2 ->
                                        reg1.add(reg2).use { regSum ->
                                            regSum.mul(regWeight).use { reg ->
                                                lossValue = lossValue.add(reg)
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    lossValue.sum().use { lossSumNd ->
                        lossSum += lossSumNd.getFloat()
                    }
                }
            }
            start = end
        }
        return lossSum / totalTriples
    }

    /**
     * Generate negative triples by replacing either the head or tail of each input triple.
     *
     * For each input triple this produces `numNegatives` candidate negatives; when `useBernoulli`
     * is true, per-relation Bernoulli probabilities bias whether the head or tail is replaced.
     * Sampled negatives avoid the original replaced entity and any existing positive triple when possible;
     * if a valid negative cannot be found within the resample cap, the original triple is returned for that slot.
     *
     * @param input NDArray of shape (batchSize, 3) containing triples as (head, relation, tail).
     * @param numEntities Total number of entities to sample from; must be greater than 1.
     * @param numNegatives Number of negative samples to generate per input triple.
     * @param useBernoulli If true, use precomputed per-relation Bernoulli probabilities to bias replacement choice.
     * @return An NDArray of shape (batchSize * numNegatives, 3) with dtype INT64 containing generated negative triples.
     * @throws IllegalArgumentException if `numEntities <= 1` or if the batch size exceeds Int indexing capacity.
     */
    private fun sampleNegatives(
        manager: NDManager,
        input: NDArray,
        numEntities: Long,
        numNegatives: Int,
        useBernoulli: Boolean,
    ): NDArray {
        require(numEntities > 1L) { "numEntities must be > 1 for negative sampling, was $numEntities." }
        val numTriples = input.size() / TRIPLE
        require(numTriples <= Int.MAX_VALUE.toLong()) {
            "Batch exceeds Int indexing capacity (numTriples=$numTriples)."
        }
        val batchCount = numTriples.toInt()
        val totalNegatives = batchCount.toLong() * numNegatives.toLong()
        val negatives = LongArray((totalNegatives * TRIPLE).toInt())
        val maxResample = NEGATIVE_RESAMPLE_CAP
        val flat = input.toLongArray()
        for (i in 0 until batchCount) {
            val base = i * TRIPLE.toInt()
            val fst = flat[base]
            val sec = flat[base + 1]
            val trd = flat[base + 2]
            val headProb = if (useBernoulli) bernoulliProb[sec] ?: 0.5f else 0.5f
            for (n in 0 until numNegatives) {
                var replaceHeadFlag = kotlin.random.Random.nextFloat() < headProb
                var ran: Long = 0L
                var checkTriplet = TripleKey(0L, 0L, 0L)
                var attempts = 0
                var found = false
                while (attempts < maxResample) {
                    ran = kotlin.random.Random.nextLong(numEntities)
                    if (replaceHeadFlag) {
                        if (ran == fst) {
                            attempts += 1
                            continue
                        }
                        checkTriplet = TripleKey(ran, sec, trd)
                    } else {
                        if (ran == trd) {
                            attempts += 1
                            continue
                        }
                        checkTriplet = TripleKey(fst, sec, ran)
                    }
                    if (!inputSet.contains(checkTriplet)) {
                        found = true
                        break
                    }
                    replaceHeadFlag = !replaceHeadFlag
                    attempts += 1
                }
                val negIndex = i * numNegatives + n
                val negBase = negIndex * TRIPLE.toInt()
                if (found) {
                    negatives[negBase] = checkTriplet.head
                    negatives[negBase + 1] = checkTriplet.rel
                    negatives[negBase + 2] = checkTriplet.tail
                } else {
                    negatives[negBase] = fst
                    negatives[negBase + 1] = sec
                    negatives[negBase + 2] = trd
                    logger.warn(
                        "Negative resample cap hit ({} attempts) for batch row {}.",
                        maxResample,
                        i,
                    )
                }
            }
        }
        return manager.create(negatives, Shape(totalNegatives, TRIPLE))
    }

    /**
     * Computes per-sample loss from positive and negative scores using either a similarity-based
     * objective or a hinge (margin) objective.
     *
     * When `useSimilarityLoss` is true, each sample's loss is
     * `log(1 + exp(-pos)) + agg(log(1 + exp(neg)))`, where `agg` is either the mean across negatives
     * or a self-adversarial weighted sum using a softmax over `neg` scaled by `SELF_ADVERSARIAL_TEMP`.
     * When `useSimilarityLoss` is false, the hinge loss `mean(max(0, pos - neg + margin))` is used
     * across negatives.
     *
     * @param pos NDArray of positive scores with shape (batchSize,).
     * @param neg NDArray of negative scores with shape (batchSize * numNegatives,) or a shape that
     *            can be aligned with `pos` when reshaped as (batchSize, numNegatives).
     * @param useSimilarityLoss If true, use the similarity-based softplus loss; otherwise use hinge loss.
     * @param useSelfAdversarial If true and `useSimilarityLoss` is true, apply self-adversarial weighting
     *                           to negative losses via a softmax over scaled negative scores.
     * @return 1D NDArray of per-sample losses with length equal to `batchSize`.
     */
    private fun computeLossFromScores(
        pos: NDArray,
        neg: NDArray,
        useSimilarityLoss: Boolean,
        useSelfAdversarial: Boolean,
    ): NDArray {
        return if (useSimilarityLoss) {
            // softplus(-pos) + softplus(neg)
            val posLoss = pos.neg().exp().add(1f).log()
            val negLoss = neg.exp().add(1f).log()
            val negAgg =
                if (useSelfAdversarial) {
                    val weights = neg.mul(SELF_ADVERSARIAL_TEMP).softmax(1)
                    weights.mul(negLoss).sum(intArrayOf(1))
                } else {
                    negLoss.mean(intArrayOf(1))
                }
            posLoss.add(negAgg)
        } else {
            val hinge = pos.expandDims(1).sub(neg).add(margin).maximum(0f)
            hinge.mean(intArrayOf(1))
        }
    }

    /**
     * Computes a Matryoshka-weighted loss by aggregating per-dimension scores from a RotatE or QuatE block.
     *
     * For each configured Matryoshka dimension that is valid for the block's entity embedding size, this function
     * computes positive and negative scores at that dimension, derives a per-dimension loss via
     * computeLossFromScores(...), multiplies it by the corresponding weight, and returns the sum across dimensions.
     *
     * @param block The model block used for scoring; must be a RotatE or QuatE instance.
     * @param sample NDArray of positive triples (shape [batchSize, 3]).
     * @param negativeSample NDArray of negative triples (shape [batchSize * numNegatives, 3]).
     * @param numNegatives Number of negative samples per positive triple.
     * @param useSimilarityLoss If true, compute similarity-based loss for each dimension; otherwise use hinge loss.
     * @param useSelfAdversarial If true and using similarity loss, apply self-adversarial weighting to negatives.
     * @return An NDArray containing the aggregated, weighted Matryoshka loss across all valid dimensions.
     *
     * @throws IllegalStateException if the provided block is not supported for Matryoshka scoring.
     * @throws IllegalArgumentException if dims and weights lengths differ or no valid Matryoshka dimension exists
     *                                  for the block's embedding size.
     */
    private fun computeMatryoshkaLoss(
        block: Any,
        sample: NDArray,
        negativeSample: NDArray,
        numNegatives: Int,
        useSimilarityLoss: Boolean,
        useSelfAdversarial: Boolean,
    ): NDArray {
        val (dims, weights, entities, edges) =
            when (block) {
                is RotatE ->
                    MatryoshkaConfig(
                        MATRYOSHKA_ROTATE_DIMS,
                        MATRYOSHKA_ROTATE_WEIGHTS,
                        block.getEntities(),
                        block.getEdges(),
                    )
                is QuatE ->
                    MatryoshkaConfig(
                        MATRYOSHKA_QUATE_DIMS,
                        MATRYOSHKA_QUATE_WEIGHTS,
                        block.getEntities(),
                        block.getEdges(),
                    )
                else -> throw IllegalStateException("Matryoshka loss used with unsupported block.")
            }
        require(dims.size == weights.size) { "Matryoshka dims and weights must have the same length." }

        val embDim = entities.shape[1]
        val usable = ArrayList<Pair<Long, Float>>(dims.size)
        for (i in dims.indices) {
            val d = dims[i]
            if (d > 0 && d <= embDim) {
                usable.add(d to weights[i])
            }
        }
        require(usable.isNotEmpty()) { "No valid Matryoshka dims for embedding size $embDim." }

        var total: NDArray? = null
        try {
            for ((dim, weight) in usable) {
                val posScores =
                    when (block) {
                        is RotatE -> block.score(sample, entities, edges, dim)
                        is QuatE -> block.score(sample, entities, edges, dim)
                        else -> throw IllegalStateException("Matryoshka loss used with unsupported block.")
                    }
                val negScores =
                    when (block) {
                        is RotatE -> block.score(negativeSample, entities, edges, dim)
                        is QuatE -> block.score(negativeSample, entities, edges, dim)
                        else -> throw IllegalStateException("Matryoshka loss used with unsupported block.")
                    }
                val negReshaped = negScores.reshape(posScores.shape[0], numNegatives.toLong())
                val loss = computeLossFromScores(posScores, negReshaped, useSimilarityLoss, useSelfAdversarial)
                val weighted = loss.mul(weight)
                loss.close()
                negReshaped.close()
                posScores.close()
                negScores.close()
                total =
                    if (total == null) {
                        weighted
                    } else {
                        val sum = total.add(weighted)
                        total.close()
                        weighted.close()
                        sum
                    }
            }
            return requireNotNull(total) { "Matryoshka loss total is null." }
        } catch (e: Exception) {
            total?.close()
            throw e
        }
    }
}

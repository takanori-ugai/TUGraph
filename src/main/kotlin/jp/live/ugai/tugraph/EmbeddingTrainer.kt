package jp.live.ugai.tugraph

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
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
                                if (useSimilarityLoss) {
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
            val validateAvg =
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
            val logEvery = maxOf(1, epoch / 500L)
            if (epochIndex % logEvery == 0L) {
                logger.debug("Epoch {} finished. (L2Loss: {})", epochIndex, trainAvg)
            }
            lossList.add(epochLoss)
            trainer.metrics?.addMetric("validate_loss", validateAvg)
            trainer.metrics?.addMetric("validate_L1Loss", validateAvg)
            trainer.notifyListeners { listener ->
                if (listener is HingeLossLoggingListener) {
                    listener.updateEvaluations(trainAvg, validateAvg)
                }
            }
            trainer.notifyListeners { listener -> listener.onEpoch(trainer) }
        }
        trainer.close()
    }

    /**
     * Releases resources held by this trainer by closing the underlying NDManager.
     *
     * After calling this method the trainer's manager and any NDArrays attached to it become invalid and should not be used.
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
        val triples = input.reshape(numTriples, TRIPLE)
        require(numTriples <= Int.MAX_VALUE.toLong()) {
            "Batch exceeds Int indexing capacity (numTriples=$numTriples)."
        }
        val batchCount = numTriples.toInt()
        val totalNegatives = batchCount.toLong() * numNegatives.toLong()
        val negatives = manager.zeros(Shape(totalNegatives, TRIPLE), DataType.INT64)
        val maxResample = NEGATIVE_RESAMPLE_CAP
        for (i in 0 until batchCount) {
            val row = triples.get(i.toLong())
            val triple = row.toLongArray()
            row.close()
            val fst = triple[0]
            val sec = triple[1]
            val trd = triple[2]
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
                if (found) {
                    negatives.setScalar(NDIndex(negIndex.toLong(), 0), checkTriplet.head)
                    negatives.setScalar(NDIndex(negIndex.toLong(), 1), checkTriplet.rel)
                    negatives.setScalar(NDIndex(negIndex.toLong(), 2), checkTriplet.tail)
                } else {
                    negatives.setScalar(NDIndex(negIndex.toLong(), 0), fst)
                    negatives.setScalar(NDIndex(negIndex.toLong(), 1), sec)
                    negatives.setScalar(NDIndex(negIndex.toLong(), 2), trd)
                    logger.warn(
                        "Negative resample cap hit ({} attempts) for batch row {}.",
                        maxResample,
                        i,
                    )
                }
            }
        }
        return negatives
    }
}
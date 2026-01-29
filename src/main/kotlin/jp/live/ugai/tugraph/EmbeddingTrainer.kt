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
    val inputList = mutableListOf<List<Long>>()
    private val inputSet: Set<List<Long>>
    private val bernoulliProb: Map<Long, Float>

    init {
        for (i in 0 until numOfTriples) {
            inputList.add(triples.get(i).toLongArray().toList())
        }
        inputSet = inputList.toHashSet()
        val relCounts = mutableMapOf<Long, Int>()
        val headSets = mutableMapOf<Long, MutableSet<Long>>()
        val tailSets = mutableMapOf<Long, MutableSet<Long>>()
        for (triple in inputList) {
            val head = triple[0]
            val rel = triple[1]
            val tail = triple[2]
            relCounts[rel] = (relCounts[rel] ?: 0) + 1
            headSets.getOrPut(rel) { mutableSetOf() }.add(head)
            tailSets.getOrPut(rel) { mutableSetOf() }.add(tail)
        }
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
     * Trains the embedding model.
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
        for (epochIndex in 1..epoch) {
            var epochLoss = 0f
            var start = 0
            while (start < numTriplesInt) {
                val end = min(start + batchSize, numTriplesInt)
                val sample = triples.get(NDIndex("$start:$end"))
                val negativeSample = sampleNegatives(sample, numOfEntities, numNegatives, useBernoulli)
                try {
                    trainer.newGradientCollector().use { gc ->
                        val f0 = trainer.forward(NDList(sample, negativeSample))
                        val pos = f0[0]
                        val neg = f0[1].reshape(pos.shape[0], numNegatives.toLong())
                        var lossValue =
                            if (useSimilarityLoss) {
                                // softplus(-pos) + softplus(neg)
                                val posLoss = pos.neg().exp().add(1f).log()
                                val negLoss = neg.exp().add(1f).log()
                                val negAgg =
                                    if (useSelfAdversarial) {
                                        val scaled = neg.mul(SELF_ADVERSARIAL_TEMP)
                                        val exp = scaled.exp()
                                        val weights = exp.div(exp.sum(intArrayOf(1), true))
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
                            val reg =
                                block.parameters.valueAt(0).array.pow(2).sum()
                                    .add(block.parameters.valueAt(1).array.pow(2).sum())
                                    .mul(regWeight)
                            lossValue = lossValue.add(reg)
                        }
                        if (!firstNanReported && logger.isWarnEnabled) {
                            val posNaN = pos.isNaN().any().getBoolean()
                            val negNaN = neg.isNaN().any().getBoolean()
                            val lossNaN = lossValue.isNaN().any().getBoolean()
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
                        val batchLoss = lossValue.mean().getFloat()
                        epochLoss += lossValue.sum().getFloat()
                        trainer.metrics?.addMetric("train_loss", batchLoss)
                        trainer.metrics?.addMetric("train_L1Loss", batchLoss)
                        gc.backward(lossValue.mean())
                    }
                } finally {
                    negativeSample.close()
                    sample.close()
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
        manager.close()
    }

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
            val sample = data.get(NDIndex("$start:$end"))
            val negativeSample = sampleNegatives(sample, numEntities, numNegatives, useBernoulli)
            try {
                val f0 = trainer.forward(NDList(sample, negativeSample))
                val pos = f0[0]
                val neg = f0[1].reshape(pos.shape[0], numNegatives.toLong())
                var lossValue =
                    if (useSimilarityLoss) {
                        val posLoss = pos.neg().exp().add(1f).log()
                        val negLoss = neg.exp().add(1f).log()
                        val negAgg =
                            if (useSelfAdversarial) {
                                val scaled = neg.mul(SELF_ADVERSARIAL_TEMP)
                                val exp = scaled.exp()
                                val weights = exp.div(exp.sum(intArrayOf(1), true))
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
                    val reg =
                        trainer.model.block.parameters.valueAt(0).array.pow(2).sum()
                            .add(trainer.model.block.parameters.valueAt(1).array.pow(2).sum())
                            .mul(regWeight)
                    lossValue = lossValue.add(reg)
                }
                lossSum += lossValue.sum().getFloat()
            } finally {
                negativeSample.close()
                sample.close()
            }
            start = end
        }
        return lossSum / totalTriples
    }

    /**
     * Generates vectorized negative samples for the given triples.
     *
     * This replaces either the head or tail for each triple using a random entity
     * in a vectorized manner. The sampled entity is guaranteed to differ from the
     * original head/tail for the replaced position.
     */
    private fun sampleNegatives(
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
        val totalNegatives = batchCount * numNegatives
        val negatives = input.manager.zeros(Shape(totalNegatives.toLong(), TRIPLE), DataType.INT64)
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
                val checkTriplet = mutableListOf(0L, 0L, 0L)
                var attempts = 0
                var found = false
                while (attempts < maxResample) {
                    ran = kotlin.random.Random.nextLong(numEntities)
                    if (replaceHeadFlag) {
                        if (ran == fst) {
                            attempts += 1
                            continue
                        }
                        checkTriplet[0] = ran
                        checkTriplet[1] = sec
                        checkTriplet[2] = trd
                    } else {
                        if (ran == trd) {
                            attempts += 1
                            continue
                        }
                        checkTriplet[0] = fst
                        checkTriplet[1] = sec
                        checkTriplet[2] = ran
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
                    negatives.setScalar(NDIndex(negIndex.toLong(), 0), checkTriplet[0])
                    negatives.setScalar(NDIndex(negIndex.toLong(), 1), checkTriplet[1])
                    negatives.setScalar(NDIndex(negIndex.toLong(), 2), checkTriplet[2])
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

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

    init {
        for (i in 0 until numOfTriples) {
            inputList.add(triples.get(i).toLongArray().toList())
        }
        inputSet = inputList.toHashSet()
    }

    /**
     * Trains the embedding model.
     */
    fun training() {
        val checkTriplet = mutableListOf(0L, 0L, 0L)
        val batchSize = maxOf(1, BATCH_SIZE)
        val numTriplesInt = numOfTriples.toInt()
        require(numTriplesInt > 0) { "No triples available for training (numOfTriples=0)." }
        var firstNanReported = false
        (1..epoch).forEach {
            var epochLoss = 0f
            var start = 0
            while (start < numTriplesInt) {
                val end = min(start + batchSize, numTriplesInt)
                val sample = triples.get(NDIndex("$start:$end"))
                val negativeSample =
                    when (val block = trainer.model.block) {
                        is TransR -> block.sampleNegatives(sample)
                        else -> sampleNegatives(sample, numOfEntities)
                    }
                trainer.newGradientCollector().use { gc ->
                    val f0 = trainer.forward(NDList(sample, negativeSample))
                    val pos = f0[0]
                    val neg = f0[1]
                    val hinge = pos.sub(neg).add(margin).maximum(0f)
                    if (!firstNanReported) {
                        val posNaN = pos.isNaN().any().getBoolean()
                        val negNaN = neg.isNaN().any().getBoolean()
                        val hingeNaN = hinge.isNaN().any().getBoolean()
                        if (posNaN || negNaN || hingeNaN) {
                            println(
                                "NaN detected in batch $start:$end " +
                                    "(pos=$posNaN, neg=$negNaN, hinge=$hingeNaN)",
                            )
                            println("Sample: $sample")
                            println("Negative sample: $negativeSample")
                            firstNanReported = true
                        }
                    }
                    val batchLoss = hinge.mean().getFloat()
                    epochLoss += hinge.sum().getFloat()
                    trainer.metrics?.addMetric("train_loss", batchLoss)
                    trainer.metrics?.addMetric("train_L1Loss", batchLoss)
                    gc.backward(hinge.mean())
                }
                trainer.step()
                when (val block = trainer.model.block) {
                    is TransE -> block.normalize()
                    is TransR -> block.normalize()
                }
                start = end
            }
            val trainAvg = epochLoss / numOfTriples
            val validateAvg = evaluateEpochLoss(triples, numTriplesInt, batchSize, numOfEntities)
            val logEvery = maxOf(1, epoch / 500L)
            if (it % logEvery == 0L) {
                logger.debug("Epoch {} finished. (L2Loss: {})", it, trainAvg)
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
    ): Float {
        var lossSum = 0f
        var start = 0
        while (start < totalTriples) {
            val end = min(start + batchSize, totalTriples)
            val sample = data.get(NDIndex("$start:$end"))
            val negativeSample = sampleNegatives(sample, numEntities)
            val f0 = trainer.forward(NDList(sample, negativeSample))
            val pos = f0[0]
            val neg = f0[1]
            val hinge = pos.sub(neg).add(margin).maximum(0f)
            lossSum += hinge.sum().getFloat()
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
    ): NDArray {
        val numTriples = input.size() / TRIPLE
        val triples = input.reshape(numTriples, TRIPLE)
        val manager = triples.manager

        val heads = triples.get(NDIndex(":, 0"))
        val relations = triples.get(NDIndex(":, 1"))
        val tails = triples.get(NDIndex(":, 2"))

        val offsetHead = manager.randomInteger(1, numEntities, Shape(numTriples), DataType.INT64)
        val offsetTail = manager.randomInteger(1, numEntities, Shape(numTriples), DataType.INT64)
        val headSum = heads.add(offsetHead)
        val tailSum = tails.add(offsetTail)
        val headWrap = headSum.gte(numEntities).toType(DataType.INT64, false)
        val tailWrap = tailSum.gte(numEntities).toType(DataType.INT64, false)
        val randHead = headSum.sub(headWrap.mul(numEntities))
        val randTail = tailSum.sub(tailWrap.mul(numEntities))

        val replaceHead = manager.randomInteger(0, 2, Shape(numTriples), DataType.INT64)
        val replaceMask = replaceHead.toType(DataType.INT64, false)
        val keepMask = replaceMask.eq(0).toType(DataType.INT64, false)

        val newHeads = heads.mul(keepMask).add(randHead.mul(replaceMask))
        val newTails = tails.mul(replaceMask).add(randTail.mul(keepMask))

        return newHeads.expandDims(1)
            .concat(relations.expandDims(1), 1)
            .concat(newTails.expandDims(1), 1)
    }
}

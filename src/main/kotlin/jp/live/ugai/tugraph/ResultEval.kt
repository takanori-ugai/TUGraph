package jp.live.ugai.tugraph

import ai.djl.inference.Predictor
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape

/**
 * A class for evaluating the results of a model.
 *
 * @property inputList The list of input triples.
 * @property manager The NDManager instance used for creating NDArrays.
 * @property predictor The Predictor instance for making predictions.
 * @property numEntities The number of entities in the data.
 */
class ResultEval(
    /** Input triples used for evaluation. */
    val inputList: List<LongArray>,
    /** NDManager used to create NDArrays for evaluation. */
    val manager: NDManager,
    /** Predictor used to score candidate triples. */
    val predictor: Predictor<NDList, NDList>,
    /** Total number of entities in the knowledge graph. */
    val numEntities: Long,
    /** Whether higher scores indicate better ranking (e.g., ComplEx). */
    val higherIsBetter: Boolean = false,
) {
    private val col0Index = NDIndex(":, 0")
    private val col1Index = NDIndex(":, 1")

    private data class EvalBatch(
        val modelInput: NDArray,
        val trueIds: IntArray,
        val batchSize: Int,
        private val closeables: List<NDArray>,
    ) : AutoCloseable {
        override fun close() {
            closeables.forEach { it.close() }
        }
    }

    private fun computeRanks(
        evalBatchSize: Int,
        buildBatch: (start: Int, end: Int) -> EvalBatch,
    ): List<Int> {
        val ranks = mutableListOf<Int>()
        var start = 0
        while (start < inputList.size) {
            val end = minOf(start + evalBatchSize, inputList.size)
            val batch = buildBatch(start, end)
            batch.use {
                val prediction = predictor.predict(NDList(batch.modelInput)).singletonOrThrow()
                try {
                    val scores = prediction.reshape(batch.batchSize.toLong(), numEntities)
                    try {
                        for (i in 0 until batch.batchSize) {
                            val itemScores = scores.get(i.toLong())
                            try {
                                val trueId = batch.trueIds[i].toLong()
                                val trueScoreNd = itemScores.get(trueId)
                                val trueScore = trueScoreNd.getFloat()
                                trueScoreNd.close()
                                val countBetterNd =
                                    if (higherIsBetter) {
                                        itemScores.gt(trueScore).sum()
                                    } else {
                                        itemScores.lt(trueScore).sum()
                                    }
                                val rank = (countBetterNd.getLong() + 1L).toInt()
                                countBetterNd.close()
                                ranks.add(rank)
                            } finally {
                                itemScores.close()
                            }
                        }
                    } finally {
                        scores.close()
                    }
                } finally {
                    prediction.close()
                }
            }
            start = end
        }
        return ranks
    }
    /**
     * Computes evaluation metrics for tail prediction.
     *
     * @return A map containing the evaluation metrics: HIT@1, HIT@10, HIT@100, and MRR.
     */
    fun getTailResult(): Map<String, Float> {
        val result = mutableMapOf<String, Float>()
        require(numEntities <= Int.MAX_VALUE.toLong()) {
            "numEntities exceeds Int.MAX_VALUE; ResultEval uses Int indexing (numEntities=$numEntities)."
        }
        val entityRange = manager.arange(0, numEntities.toInt(), 1, DataType.INT64).reshape(numEntities, 1)
        val evalBatchSize = maxOf(1, RESULT_EVAL_BATCH_SIZE)

        val ranks =
            computeRanks(evalBatchSize) { start, end ->
                val batchSize = end - start
                val headArr = LongArray(batchSize)
                val relArr = LongArray(batchSize)
                val tailIds = IntArray(batchSize)
                for (i in 0 until batchSize) {
                    val triple = inputList[start + i]
                    headArr[i] = triple[0]
                    relArr[i] = triple[1]
                    val tailId = triple[2].toInt()
                    require(tailId in 0 until numEntities.toInt()) {
                        "Tail id out of range: $tailId (numEntities=$numEntities)."
                    }
                    tailIds[i] = tailId
                }
                val headRel = manager.zeros(Shape(batchSize.toLong(), 2), DataType.INT64)
                val headCol = manager.create(headArr)
                val relCol = manager.create(relArr)
                headRel.set(col0Index, headCol)
                headRel.set(col1Index, relCol)
                val repeatedHeadRel =
                    headRel.expandDims(1).repeat(1, numEntities).reshape(batchSize.toLong() * numEntities, 2)
                val entityRangeRepeated = entityRange.repeat(0, batchSize.toLong())
                val modelInput = repeatedHeadRel.concat(entityRangeRepeated, 1)
                EvalBatch(
                    modelInput = modelInput,
                    trueIds = tailIds,
                    batchSize = batchSize,
                    closeables = listOf(modelInput, entityRangeRepeated, repeatedHeadRel, headRel, headCol, relCol),
                )
            }

        // Compute metrics
        val total = ranks.size.toFloat()
        result["HIT@1"] = ranks.count { it <= 1 } / total
        result["HIT@10"] = ranks.count { it <= 10 } / total
        result["HIT@100"] = ranks.count { it <= 100 } / total
        result["MRR"] = ranks.sumOf { 1.0 / it }.toFloat() / total

        entityRange.close()
        return result
    }

    /**
     * Computes evaluation metrics for head prediction.
     *
     * @return A map containing the evaluation metrics: HIT@1, HIT@10, HIT@100, and MRR.
     */
    fun getHeadResult(): Map<String, Float> {
        val result = mutableMapOf<String, Float>()
        require(numEntities <= Int.MAX_VALUE.toLong()) {
            "numEntities exceeds Int.MAX_VALUE; ResultEval uses Int indexing (numEntities=$numEntities)."
        }
        val entityRange = manager.arange(0, numEntities.toInt(), 1, DataType.INT64).reshape(numEntities, 1)
        val evalBatchSize = maxOf(1, RESULT_EVAL_BATCH_SIZE)

        val ranks =
            computeRanks(evalBatchSize) { start, end ->
                val batchSize = end - start
                val relArr = LongArray(batchSize)
                val tailArr = LongArray(batchSize)
                val headIds = IntArray(batchSize)
                for (i in 0 until batchSize) {
                    val triple = inputList[start + i]
                    relArr[i] = triple[1]
                    tailArr[i] = triple[2]
                    val headId = triple[0].toInt()
                    require(headId in 0 until numEntities.toInt()) {
                        "Head id out of range: $headId (numEntities=$numEntities)."
                    }
                    headIds[i] = headId
                }
                val relTail = manager.zeros(Shape(batchSize.toLong(), 2), DataType.INT64)
                val relCol = manager.create(relArr)
                val tailCol = manager.create(tailArr)
                relTail.set(col0Index, relCol)
                relTail.set(col1Index, tailCol)
                val repeatedRelTail =
                    relTail.expandDims(1).repeat(1, numEntities).reshape(batchSize.toLong() * numEntities, 2)
                val entityRangeRepeated = entityRange.repeat(0, batchSize.toLong())
                val modelInput = entityRangeRepeated.concat(repeatedRelTail, 1)
                EvalBatch(
                    modelInput = modelInput,
                    trueIds = headIds,
                    batchSize = batchSize,
                    closeables = listOf(modelInput, entityRangeRepeated, repeatedRelTail, relTail, relCol, tailCol),
                )
            }

        // Compute metrics
        val total = ranks.size.toFloat()
        result["HIT@1"] = ranks.count { it <= 1 } / total
        result["HIT@10"] = ranks.count { it <= 10 } / total
        result["HIT@100"] = ranks.count { it <= 100 } / total
        result["MRR"] = ranks.sumOf { 1.0 / it }.toFloat() / total

        entityRange.close()
        return result
    }

    /**
     * Closes the NDManager instance.
     */
    fun close() {
        manager.close()
    }
}

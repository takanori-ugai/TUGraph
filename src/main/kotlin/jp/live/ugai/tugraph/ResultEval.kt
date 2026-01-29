package jp.live.ugai.tugraph

import ai.djl.inference.Predictor
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
) {
    private val col0Index = NDIndex(":, 0")
    private val col1Index = NDIndex(":, 1")
    /**
     * Computes evaluation metrics for tail prediction.
     *
     * @return A map containing the evaluation metrics: HIT@1, HIT@10, HIT@100, and MRR.
     */
    fun getTailResult(): Map<String, Float> {
        val result = mutableMapOf<String, Float>()
        val ranks = mutableListOf<Int>()
        val entityRange = manager.arange(0, numEntities.toInt(), 1, DataType.INT64).reshape(numEntities, 1)
        val evalBatchSize = maxOf(1, RESULT_EVAL_BATCH_SIZE)

        var start = 0
        while (start < inputList.size) {
            val end = minOf(start + evalBatchSize, inputList.size)
            val batchSize = end - start
            val headArr = LongArray(batchSize)
            val relArr = LongArray(batchSize)
            val tailIds = IntArray(batchSize)
            for (i in 0 until batchSize) {
                val triple = inputList[start + i]
                headArr[i] = triple[0]
                relArr[i] = triple[1]
                tailIds[i] = triple[2].toInt()
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
            try {
                val prediction = predictor.predict(NDList(modelInput)).singletonOrThrow()
                val scores = prediction.toFloatArray()
                prediction.close()
                for (i in 0 until batchSize) {
                    val base = i * numEntities.toInt()
                    val trueTail = tailIds[i]
                    val trueScore = scores[base + trueTail]
                    var rank = 1
                    val endIdx = base + numEntities.toInt()
                    for (j in base until endIdx) {
                        if (scores[j] < trueScore) {
                            rank += 1
                        }
                    }
                    ranks.add(rank)
                }
            } finally {
                modelInput.close()
                entityRangeRepeated.close()
                repeatedHeadRel.close()
                headRel.close()
                headCol.close()
                relCol.close()
            }
            start = end
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
        val ranks = mutableListOf<Int>()
        val entityRange = manager.arange(0, numEntities.toInt(), 1, DataType.INT64).reshape(numEntities, 1)
        val evalBatchSize = maxOf(1, RESULT_EVAL_BATCH_SIZE)

        var start = 0
        while (start < inputList.size) {
            val end = minOf(start + evalBatchSize, inputList.size)
            val batchSize = end - start
            val relArr = LongArray(batchSize)
            val tailArr = LongArray(batchSize)
            val headIds = IntArray(batchSize)
            for (i in 0 until batchSize) {
                val triple = inputList[start + i]
                relArr[i] = triple[1]
                tailArr[i] = triple[2]
                headIds[i] = triple[0].toInt()
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
            try {
                val prediction = predictor.predict(NDList(modelInput)).singletonOrThrow()
                val scores = prediction.toFloatArray()
                prediction.close()
                for (i in 0 until batchSize) {
                    val base = i * numEntities.toInt()
                    val trueHead = headIds[i]
                    val trueScore = scores[base + trueHead]
                    var rank = 1
                    val endIdx = base + numEntities.toInt()
                    for (j in base until endIdx) {
                        if (scores[j] < trueScore) {
                            rank += 1
                        }
                    }
                    ranks.add(rank)
                }
            } finally {
                modelInput.close()
                entityRangeRepeated.close()
                repeatedRelTail.close()
                relTail.close()
                relCol.close()
                tailCol.close()
            }
            start = end
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

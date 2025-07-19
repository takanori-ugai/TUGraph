package jp.live.ugai.tugraph

import ai.djl.inference.Predictor
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType

/**
 * A class for evaluating the results of a model.
 *
 * @property inputList The list of input triples.
 * @property manager The NDManager instance used for creating NDArrays.
 * @property predictor The Predictor instance for making predictions.
 */
class ResultEval(val inputList: List<LongArray>, val manager: NDManager, val predictor: Predictor<NDList, NDList>) {
    /**
     * Computes evaluation metrics for tail prediction.
     *
     * @return A map containing the evaluation metrics: HIT@1, HIT@10, HIT@100, and MRR.
     */
    fun getTailResult(): Map<String, Float> {
        val result = mutableMapOf<String, Float>()
        val ranks = mutableListOf<Int>()

        inputList.forEach { triple ->
            // Prepare input for prediction
            val head = triple[0]
            val relation = triple[1]
            val tail = triple[2].toInt()

            val repeatedInput =
                manager.create(longArrayOf(head, relation))
                    .reshape(1, 2)
                    .repeat(0, NUM_ENTITIES)
            val entityRange = manager.arange(0, NUM_ENTITIES.toInt(), 1, DataType.INT64).reshape(NUM_ENTITIES, 1)
            val modelInput = repeatedInput.concat(entityRange, 1)

            // Predict scores for all possible tails
            val prediction = predictor.predict(NDList(modelInput)).singletonOrThrow()
            val scores = prediction.toFloatArray()
            prediction.close()

            // Rank the true tail among all candidates
            val sortedIndices = scores.withIndex().sortedBy { it.value }.map { it.index }
            val rank = sortedIndices.indexOf(tail) + 1
            ranks.add(rank)
        }

        // Compute metrics
        val total = ranks.size.toFloat()
        result["HIT@1"] = ranks.count { it <= 1 } / total
        result["HIT@10"] = ranks.count { it <= 10 } / total
        result["HIT@100"] = ranks.count { it <= 100 } / total
        result["MRR"] = ranks.sumOf { 1.0 / it }.toFloat() / total

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

        inputList.forEach { triple ->
            // Compute scores for all possible head entities
            val scores =
                (0 until NUM_ENTITIES).map { head ->
                    val input = manager.create(longArrayOf(head.toLong(), triple[1], triple[2]))
                    val score = predictor.predict(NDList(input)).singletonOrThrow().toFloatArray()[0]
                    score
                }

            // Rank the true head among all candidates
            val sortedIndices = scores.withIndex().sortedBy { it.value }.map { it.index }
            val trueHead = triple[0].toInt()
            val rank = sortedIndices.indexOf(trueHead) + 1
            ranks.add(rank)
        }

        // Compute metrics
        val total = ranks.size.toFloat()
        result["HIT@1"] = ranks.count { it <= 1 } / total
        result["HIT@10"] = ranks.count { it <= 10 } / total
        result["HIT@100"] = ranks.count { it <= 100 } / total
        result["MRR"] = ranks.sumOf { 1.0 / it }.toFloat() / total

        return result
    }

    /**
     * Closes the NDManager instance.
     */
    fun close() {
        manager.close()
    }
}

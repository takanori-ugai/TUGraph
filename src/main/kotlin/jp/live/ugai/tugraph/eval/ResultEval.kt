package jp.live.ugai.tugraph.eval

import ai.djl.inference.Predictor
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import jp.live.ugai.tugraph.RESULT_EVAL_BATCH_SIZE
import jp.live.ugai.tugraph.RESULT_EVAL_ENTITY_CHUNK_SIZE

/**
 * A class for evaluating the results of a model.
 *
 * @property inputList The list of input triples.
 * @property manager The NDManager instance used for creating NDArrays.
 * @property predictor The Predictor instance for making predictions.
 * @property numEntities The number of entities in the data.
 */
open class ResultEval(
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
    /** Whether to perform filtered evaluation by removing other true triples from candidate sets. */
    val filtered: Boolean = false,
) : AutoCloseable {
    private val col0Index = NDIndex(":, 0")
    private val col1Index = NDIndex(":, 1")

    private data class PairKey(
        val a: Long,
        val b: Long,
    )

    private val tailFilter: Map<PairKey, IntArray> by lazy {
        val map = mutableMapOf<PairKey, MutableSet<Int>>()
        for (triple in inputList) {
            val head = triple[0]
            val rel = triple[1]
            val tail = triple[2].toInt()
            map.getOrPut(PairKey(head, rel)) { mutableSetOf() }.add(tail)
        }
        map.mapValues { (_, v) -> v.toIntArray() }
    }

    private val headFilter: Map<PairKey, IntArray> by lazy {
        val map = mutableMapOf<PairKey, MutableSet<Int>>()
        for (triple in inputList) {
            val rel = triple[1]
            val tail = triple[2]
            val head = triple[0].toInt()
            map.getOrPut(PairKey(rel, tail)) { mutableSetOf() }.add(head)
        }
        map.mapValues { (_, v) -> v.toIntArray() }
    }

    /**
     * Adjusts tail-prediction ranks using filtered evaluation by removing competing true tails.
     *
     * If `filtered` is false the input `rawRanks` are returned unchanged. When filtered evaluation
     * is enabled, for each input triple this subtracts the number of other true tail candidates
     * (from `tailFilter`) that score better than the true tail, clamping each adjusted rank to a
     * minimum of 1.
     *
     * @param rawRanks Per-example raw ranks for tail predictions.
     * @return An IntArray of adjusted ranks where each rank is decreased by the count of other true
     * tails that outperform the true tail, with a minimum value of 1 for every rank.
     */
    private fun applyFilteredTail(rawRanks: IntArray): IntArray {
        if (!filtered) return rawRanks
        val adjusted = rawRanks.copyOf()
        for (i in inputList.indices) {
            val triple = inputList[i]
            val head = triple[0]
            val rel = triple[1]
            val tail = triple[2].toInt()
            val candidates = tailFilter[PairKey(head, rel)] ?: continue
            if (candidates.size <= 1) continue
            manager.newSubManager().use { sm ->
                val trueInput = sm.create(longArrayOf(head, rel, tail.toLong())).reshape(1, 3)
                val trueScore = predictor.predict(NDList(trueInput)).singletonOrThrow()
                val trueVal = trueScore.getFloat()
                trueScore.close()
                trueInput.close()
                var count = 0
                val filteredCount = candidates.count { it != tail }
                if (filteredCount > 0) {
                    val data = LongArray(filteredCount * 3)
                    var idx = 0
                    for (cand in candidates) {
                        if (cand == tail) continue
                        data[idx] = head
                        data[idx + 1] = rel
                        data[idx + 2] = cand.toLong()
                        idx += 3
                    }
                    val candInput = sm.create(data, Shape(filteredCount.toLong(), 3))
                    val candScores = predictor.predict(NDList(candInput)).singletonOrThrow()
                    val scores = candScores.toFloatArray()
                    for (score in scores) {
                        if (higherIsBetter) {
                            if (score > trueVal) count++
                        } else {
                            if (score < trueVal) count++
                        }
                    }
                    candScores.close()
                    candInput.close()
                }
                adjusted[i] = maxOf(1, adjusted[i] - count)
            }
        }
        return adjusted
    }

    /**
     * Adjusts head-prediction ranks by removing competing true heads when filtered evaluation is enabled.
     *
     * When `filtered` is false, the input `rawRanks` array is returned unchanged. Otherwise, for each
     * input triple this computes how many true candidate heads (from `headFilter`) score better than
     * the true head and subtracts that count from the corresponding raw rank, with a minimum rank of 1.
     *
     * @param rawRanks Per-example raw ranks for head prediction, aligned with `inputList`.
     * @return A new `IntArray` of adjusted ranks aligned with `inputList`, where each value is at least 1.
     */
    private fun applyFilteredHead(rawRanks: IntArray): IntArray {
        if (!filtered) return rawRanks
        val adjusted = rawRanks.copyOf()
        for (i in inputList.indices) {
            val triple = inputList[i]
            val head = triple[0].toInt()
            val rel = triple[1]
            val tail = triple[2]
            val candidates = headFilter[PairKey(rel, tail)] ?: continue
            if (candidates.size <= 1) continue
            manager.newSubManager().use { sm ->
                val trueInput = sm.create(longArrayOf(head.toLong(), rel, tail)).reshape(1, 3)
                val trueScore = predictor.predict(NDList(trueInput)).singletonOrThrow()
                val trueVal = trueScore.getFloat()
                trueScore.close()
                trueInput.close()
                var count = 0
                val filteredCount = candidates.count { it != head }
                if (filteredCount > 0) {
                    val data = LongArray(filteredCount * 3)
                    var idx = 0
                    for (cand in candidates) {
                        if (cand == head) continue
                        data[idx] = cand.toLong()
                        data[idx + 1] = rel
                        data[idx + 2] = tail
                        idx += 3
                    }
                    val candInput = sm.create(data, Shape(filteredCount.toLong(), 3))
                    val candScores = predictor.predict(NDList(candInput)).singletonOrThrow()
                    val scores = candScores.toFloatArray()
                    for (score in scores) {
                        if (higherIsBetter) {
                            if (score > trueVal) count++
                        } else {
                            if (score < trueVal) count++
                        }
                    }
                    candScores.close()
                    candInput.close()
                }
                adjusted[i] = maxOf(1, adjusted[i] - count)
            }
        }
        return adjusted
    }

    protected data class EvalBatch(
        val basePair: NDArray,
        val trueIds: IntArray,
        val trueEntityCol: NDArray,
        val batchSize: Int,
        private val closeables: List<NDArray>,
    ) : AutoCloseable {
        /**
         * Closes all managed resources.
         *
         * Calls `close()` on each entry in `closeables`.
         */
        override fun close() {
            closeables.forEach { it.close() }
        }
    }

    protected data class ChunkInput(
        val modelInput: NDArray,
        private val closeables: List<NDArray>,
    ) : AutoCloseable {
        /**
         * Closes all managed resources.
         *
         * Calls `close()` on each entry in `closeables`.
         */
        override fun close() {
            closeables.forEach { it.close() }
        }
    }

    protected enum class EvalMode {
        TAIL,
        HEAD,
    }

    /**
     * Compute per-example ranks for the configured evaluation mode.
     *
     * The default implementation constructs model inputs for either tail or head prediction,
     * scores true and candidate entities in memory-efficient chunks, and returns a rank per input example.
     *
     * @param evalBatchSize Maximum number of input triples processed in an outer batch.
     * @param entityChunkSize Maximum number of candidate entities processed per inner chunk.
     * @param mode Evaluation mode determining whether to predict tails or heads.
     * @param buildBatch Function that builds an EvalBatch for input indices in [start, end).
     * @return An IntArray of ranks for each evaluated example where `1` indicates the top rank (best).
     */
    protected open fun computeRanks(
        evalBatchSize: Int,
        entityChunkSize: Int,
        mode: EvalMode,
        buildBatch: (start: Int, end: Int) -> EvalBatch,
    ): IntArray =
        when (mode) {
            EvalMode.TAIL ->
                computeRanksChunked(
                    evalBatchSize,
                    entityChunkSize,
                    buildBatch = buildBatch,
                    buildTrueInput = { basePair, trueEntityCol ->
                        basePair.concat(trueEntityCol, 1)
                    },
                    buildChunkInput = { basePair, entityChunk ->
                        val batchSize = basePair.shape[0]
                        val chunkSize = entityChunk.shape[0]
                        val repeatedHeadRel =
                            basePair.expandDims(1).repeat(1, chunkSize).reshape(batchSize * chunkSize, 2)
                        val entityRangeRepeated = entityChunk.repeat(0, batchSize)
                        val modelInput = repeatedHeadRel.concat(entityRangeRepeated, 1)
                        ChunkInput(
                            modelInput = modelInput,
                            closeables = listOf(modelInput, entityRangeRepeated, repeatedHeadRel),
                        )
                    },
                )
            EvalMode.HEAD ->
                computeRanksChunked(
                    evalBatchSize,
                    entityChunkSize,
                    buildBatch = buildBatch,
                    buildTrueInput = { basePair, trueEntityCol ->
                        trueEntityCol.concat(basePair, 1)
                    },
                    buildChunkInput = { basePair, entityChunk ->
                        val batchSize = basePair.shape[0]
                        val chunkSize = entityChunk.shape[0]
                        val repeatedRelTail =
                            basePair.expandDims(1).repeat(1, chunkSize).reshape(batchSize * chunkSize, 2)
                        val entityRangeRepeated = entityChunk.repeat(0, batchSize)
                        val modelInput = entityRangeRepeated.concat(repeatedRelTail, 1)
                        ChunkInput(
                            modelInput = modelInput,
                            closeables = listOf(modelInput, entityRangeRepeated, repeatedRelTail),
                        )
                    },
                )
        }

    /**
     * Computes the rank of the true entity for each input triple by comparing its score
     * against scores of all candidate entities in chunks.
     *
     * The rank for an item is 1 plus the number of candidate entities whose score is strictly
     * better than the true entity's score. "Strictly better" means greater than the true score
     * when `higherIsBetter` is true, or less than the true score when `higherIsBetter` is false.
     *
     * @param evalBatchSize Maximum number of input triples processed per outer batch.
     * @param entityChunkSize Maximum number of candidate entities evaluated per inner chunk.
     * @param buildBatch Builds an EvalBatch for the half-open range [start, end).
     * @param buildTrueInput Constructs the model input used to compute the true-entity scores from a base pair and the column of true entity IDs.
     * @param buildChunkInput Constructs the per-chunk model input from a base pair and a column of candidate entity IDs.
     * @return A list of ranks (one Int per input triple, in inputList order).
     */
    protected fun computeRanksChunked(
        evalBatchSize: Int,
        entityChunkSize: Int,
        buildBatch: (start: Int, end: Int) -> EvalBatch,
        buildTrueInput: (basePair: NDArray, trueEntityCol: NDArray) -> NDArray,
        buildChunkInput: (basePair: NDArray, entityChunk: NDArray) -> ChunkInput,
    ): IntArray {
        val totalSize = inputList.size
        if (totalSize == 0) {
            return IntArray(0)
        }
        val ranks = IntArray(totalSize)
        var outIndex = 0
        val numEntitiesInt = numEntities.toInt()
        var start = 0
        while (start < totalSize) {
            val end = minOf(start + evalBatchSize, totalSize)
            val batch = buildBatch(start, end)
            batch.use {
                var trueInput: NDArray? = null
                var trueScores: NDArray? = null
                var trueScores2d: NDArray? = null
                var countBetter: NDArray? = null
                try {
                    trueInput = buildTrueInput(batch.basePair, batch.trueEntityCol)
                    trueScores = predictor.predict(NDList(trueInput)).singletonOrThrow()
                    trueScores2d = trueScores.reshape(batch.batchSize.toLong(), 1)
                    countBetter = manager.zeros(Shape(batch.batchSize.toLong()), DataType.INT64)
                    val chunkSize = maxOf(1, entityChunkSize)
                    var chunkStart = 0
                    while (chunkStart < numEntitiesInt) {
                        val chunkEnd = minOf(chunkStart + chunkSize, numEntitiesInt)
                        val entityChunk =
                            manager
                                .arange(chunkStart, chunkEnd, 1, DataType.INT64)
                                .reshape((chunkEnd - chunkStart).toLong(), 1)
                        val chunkInput = buildChunkInput(batch.basePair, entityChunk)
                        chunkInput.use {
                            val prediction = predictor.predict(NDList(chunkInput.modelInput)).singletonOrThrow()
                            val scores =
                                prediction.reshape(
                                    batch.batchSize.toLong(),
                                    (chunkEnd - chunkStart).toLong(),
                                )
                            val countBetterNd =
                                if (higherIsBetter) {
                                    scores.gt(trueScores2d).sum(intArrayOf(1))
                                } else {
                                    scores.lt(trueScores2d).sum(intArrayOf(1))
                                }
                            try {
                                countBetter.addi(countBetterNd)
                            } finally {
                                countBetterNd.close()
                                scores.close()
                                prediction.close()
                            }
                        }
                        entityChunk.close()
                        chunkStart = chunkEnd
                    }
                    val ranksNd = countBetter.add(1)
                    try {
                        val batchRanks = ranksNd.toLongArray()
                        for (rank in batchRanks) {
                            ranks[outIndex++] = rank.toInt()
                        }
                    } finally {
                        ranksNd.close()
                    }
                } finally {
                    countBetter?.close()
                    trueScores2d?.close()
                    trueScores?.close()
                    trueInput?.close()
                }
            }
            start = end
        }
        return if (outIndex == totalSize) ranks else ranks.copyOf(outIndex)
    }

    /**
     * Compute standard link-prediction metrics from a list of ranks.
     *
     * @param ranks List of 1-based ranks where each element is the rank of the true entity for a query.
     * @return A map with keys:
     *  - "HIT@1": proportion of ranks <= 1,
     *  - "HIT@10": proportion of ranks <= 10,
     *  - "HIT@100": proportion of ranks <= 100,
     *  - "MRR": mean reciprocal rank (mean of 1 / rank).
     */
    private fun computeMetrics(ranks: IntArray): Map<String, Float> {
        val total = ranks.size.toFloat()
        if (total == 0f) {
            return mapOf(
                "HIT@1" to 0f,
                "HIT@10" to 0f,
                "HIT@100" to 0f,
                "MRR" to 0f,
            )
        }
        var hit1 = 0
        var hit10 = 0
        var hit100 = 0
        var mrr = 0.0
        for (rank in ranks) {
            if (rank <= 1) hit1++
            if (rank <= 10) hit10++
            if (rank <= 100) hit100++
            mrr += 1.0 / rank
        }
        return mapOf(
            "HIT@1" to hit1 / total,
            "HIT@10" to hit10 / total,
            "HIT@100" to hit100 / total,
            "MRR" to (mrr / total).toFloat(),
        )
    }

    /**
     * Compute standard evaluation metrics for tail (object) prediction over the evaluator's input triples.
     *
     * If filtered evaluation is enabled on this ResultEval instance, metrics are computed using filtered ranks.
     *
     * @return A map with keys "HIT@1", "HIT@10", "HIT@100", and "MRR". "HIT@k" is the proportion of examples whose rank is less than or equal to k; "MRR" is the mean reciprocal rank (mean of 1/rank).
     */
    fun getTailResult(): Map<String, Float> {
        require(numEntities <= Int.MAX_VALUE.toLong()) {
            "numEntities exceeds Int.MAX_VALUE; ResultEval uses Int indexing (numEntities=$numEntities)."
        }
        val evalBatchSize = maxOf(1, RESULT_EVAL_BATCH_SIZE)
        val entityChunkSize = maxOf(1, RESULT_EVAL_ENTITY_CHUNK_SIZE)

        val buildBatch = { start: Int, end: Int ->
            val batchSize = end - start
            val headArr = LongArray(batchSize)
            val relArr = LongArray(batchSize)
            val tailIds = IntArray(batchSize)
            for (i in 0 until batchSize) {
                val triple = inputList[start + i]
                val headId = triple[0].toInt()
                require(headId in 0 until numEntities.toInt()) {
                    "Head id out of range: $headId (numEntities=$numEntities)."
                }
                headArr[i] = headId.toLong()
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
            val trueTailCol =
                manager
                    .create(LongArray(batchSize) { tailIds[it].toLong() })
                    .reshape(batchSize.toLong(), 1)
            EvalBatch(
                basePair = headRel,
                trueIds = tailIds,
                trueEntityCol = trueTailCol,
                batchSize = batchSize,
                closeables = listOf(trueTailCol, headRel, headCol, relCol),
            )
        }

        val ranks =
            computeRanks(
                evalBatchSize,
                entityChunkSize,
                EvalMode.TAIL,
                buildBatch,
            )

        val finalRanks = applyFilteredTail(ranks)
        return computeMetrics(finalRanks)
    }

    /**
     * Computes evaluation metrics for head prediction.
     *
     * @return A map containing the evaluation metrics: HIT@1, HIT@10, HIT@100, and MRR.
     */
    fun getHeadResult(): Map<String, Float> {
        require(numEntities <= Int.MAX_VALUE.toLong()) {
            "numEntities exceeds Int.MAX_VALUE; ResultEval uses Int indexing (numEntities=$numEntities)."
        }
        val evalBatchSize = maxOf(1, RESULT_EVAL_BATCH_SIZE)
        val entityChunkSize = maxOf(1, RESULT_EVAL_ENTITY_CHUNK_SIZE)

        val buildBatch = { start: Int, end: Int ->
            val batchSize = end - start
            val relArr = LongArray(batchSize)
            val tailArr = LongArray(batchSize)
            val headIds = IntArray(batchSize)
            for (i in 0 until batchSize) {
                val triple = inputList[start + i]
                relArr[i] = triple[1]
                val tailId = triple[2].toInt()
                require(tailId in 0 until numEntities.toInt()) {
                    "Tail id out of range: $tailId (numEntities=$numEntities)."
                }
                tailArr[i] = tailId.toLong()
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
            val trueHeadCol =
                manager
                    .create(LongArray(batchSize) { headIds[it].toLong() })
                    .reshape(batchSize.toLong(), 1)
            EvalBatch(
                basePair = relTail,
                trueIds = headIds,
                trueEntityCol = trueHeadCol,
                batchSize = batchSize,
                closeables = listOf(trueHeadCol, relTail, relCol, tailCol),
            )
        }

        val ranks =
            computeRanks(
                evalBatchSize,
                entityChunkSize,
                EvalMode.HEAD,
                buildBatch,
            )

        val finalRanks = applyFilteredHead(ranks)
        return computeMetrics(finalRanks)
    }

    /**
     * Releases resources held by this evaluator by closing its NDManager.
     */
    override fun close() {
        manager.close()
    }
}

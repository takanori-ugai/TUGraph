package jp.live.ugai.tugraph.eval

import ai.djl.inference.Predictor
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import jp.live.ugai.tugraph.DistMult

class ResultEvalDistMult(
    inputList: List<LongArray>,
    manager: NDManager,
    predictor: Predictor<NDList, NDList>,
    numEntities: Long,
    val distMult: DistMult,
    higherIsBetter: Boolean = true,
    filtered: Boolean = false,
) : ResultEval(inputList, manager, predictor, numEntities, higherIsBetter, filtered) {
    private val col0Index = NDIndex(":, 0")
    private val col1Index = NDIndex(":, 1")

    /**
     * Computes per-input ranks by comparing true entity scores against all entity scores in chunks.
     *
     * @param evalBatchSize Number of input rows processed per evaluation batch.
     * @param entityChunkSize Maximum number of entities loaded into memory at once when scoring.
     * @param mode Determines which column (head or tail) is treated as the fixed entity for scoring.
     * @param buildBatch Function that builds an EvalBatch for inputs in the half-open range [start, end).
     * @return An IntArray of length equal to the number of processed inputs (or fewer if truncated) where each entry is the rank (1 + count of entities with a better score than the true entity) for the corresponding input.
     */
    protected override fun computeRanks(
        evalBatchSize: Int,
        entityChunkSize: Int,
        mode: EvalMode,
        buildBatch: (start: Int, end: Int) -> EvalBatch,
    ): IntArray {
        val totalSize = inputList.size
        if (totalSize == 0) {
            return IntArray(0)
        }
        val ranks = IntArray(totalSize)
        var outIndex = 0
        // These are live parameter arrays; do not close them here.
        val entities = distMult.getEntities()
        val edges = distMult.getEdges()
        val chunkSize = maxOf(1, entityChunkSize)
        val numEntitiesInt = numEntities.toInt()
        var start = 0
        while (start < totalSize) {
            val end = minOf(start + evalBatchSize, totalSize)
            val batch = buildBatch(start, end)
            batch.use {
                val basePair = batch.basePair
                val trueEntityCol = batch.trueEntityCol
                val batchSize = batch.batchSize
                manager.newSubManager().use { batchManager ->
                    val baseCol0 = basePair.get(col0Index).also { it.attach(batchManager) }
                    val baseCol1 = basePair.get(col1Index).also { it.attach(batchManager) }
                    val trueIdsFlat =
                        trueEntityCol.reshape(batchSize.toLong()).also { it.attach(batchManager) }
                    val relIds = if (mode == EvalMode.TAIL) baseCol1 else baseCol0
                    val fixedIds = if (mode == EvalMode.TAIL) baseCol0 else baseCol1
                    val rels = edges.get(relIds).also { it.attach(batchManager) }
                    val fixed = entities.get(fixedIds).also { it.attach(batchManager) }
                    val baseMul = fixed.mul(rels).also { it.attach(batchManager) }
                    val trueEmb = entities.get(trueIdsFlat).also { it.attach(batchManager) }
                    val trueScore =
                        baseMul
                            .mul(trueEmb)
                            .sum(intArrayOf(1))
                            .reshape(batchSize.toLong(), 1)
                            .also { it.attach(batchManager) }
                    val countBetter = batchManager.zeros(Shape(batchSize.toLong()), DataType.INT64)
                    var chunkStart = 0
                    while (chunkStart < numEntitiesInt) {
                        val chunkEnd = minOf(chunkStart + chunkSize, numEntitiesInt)
                        entities.get(NDIndex("$chunkStart:$chunkEnd, :")).use { entityChunk ->
                            entityChunk.transpose().use { entityChunkT ->
                                val scores = baseMul.matMul(entityChunkT)
                                scores.use { sc ->
                                    val countBetterNd =
                                        if (higherIsBetter) {
                                            sc.gt(trueScore).sum(intArrayOf(1))
                                        } else {
                                            sc.lt(trueScore).sum(intArrayOf(1))
                                        }
                                    countBetter.addi(countBetterNd)
                                    countBetterNd.close()
                                }
                            }
                        }
                        chunkStart = chunkEnd
                    }
                    val ranksNd = countBetter.add(1)
                    val batchRanks = ranksNd.toLongArray()
                    ranksNd.close()
                    for (rank in batchRanks) {
                        ranks[outIndex++] = rank.toInt()
                    }
                }
            }
            start = end
        }
        return if (outIndex == totalSize) ranks else ranks.copyOf(outIndex)
    }
}
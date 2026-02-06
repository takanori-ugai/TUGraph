package jp.live.ugai.tugraph.eval

import ai.djl.inference.Predictor
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import jp.live.ugai.tugraph.*

class ResultEvalQuatE(
    inputList: List<LongArray>,
    manager: NDManager,
    predictor: Predictor<NDList, NDList>,
    numEntities: Long,
    val quatE: QuatE,
    higherIsBetter: Boolean = true,
    filtered: Boolean = false,
) : ResultEval(inputList, manager, predictor, numEntities, higherIsBetter, filtered) {
    private val col0Index = NDIndex(":, 0")
    private val col1Index = NDIndex(":, 1")

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
        val entities = quatE.getEntities()
        val edges = quatE.getEdges()
        val embDim4 = entities.shape[1].toInt()
        require(embDim4 % 4 == 0) { "QuatE embeddings must have dimension divisible by 4 (found $embDim4)." }
        val quarterDim = embDim4 / 4
        val rIndex = NDIndex(":, 0:$quarterDim")
        val iIndex = NDIndex(":, $quarterDim:${quarterDim * 2}")
        val jIndex = NDIndex(":, ${quarterDim * 2}:${quarterDim * 3}")
        val kIndex = NDIndex(":, ${quarterDim * 3}:")
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
                    val rel = EvalQuatE.sliceQuat(rels, rIndex, iIndex, jIndex, kIndex, batchManager)
                    val a =
                        if (mode == EvalMode.TAIL) {
                            val fixedParts =
                                EvalQuatE.sliceQuat(fixed, rIndex, iIndex, jIndex, kIndex, batchManager)
                            EvalQuatE.computeATail(fixedParts, rel, batchManager)
                        } else {
                            EvalQuatE.computeAHead(
                                rel,
                                fixedIds,
                                entities,
                                rIndex,
                                iIndex,
                                jIndex,
                                kIndex,
                                batchManager,
                            )
                        }
                    val trueScore =
                        if (mode == EvalMode.TAIL) {
                            EvalQuatE.computeTrueScoreTail(
                                a,
                                trueIdsFlat,
                                entities,
                                rIndex,
                                iIndex,
                                jIndex,
                                kIndex,
                                batchSize,
                                batchManager,
                            )
                        } else {
                            EvalQuatE.computeTrueScoreHead(
                                a,
                                trueIdsFlat,
                                entities,
                                rIndex,
                                iIndex,
                                jIndex,
                                kIndex,
                                batchSize,
                                batchManager,
                            )
                        }
                    val countBetter = batchManager.zeros(Shape(batchSize.toLong()), DataType.INT64)
                    EvalQuatE.accumulateCountBetter(
                        entities,
                        rIndex,
                        iIndex,
                        jIndex,
                        kIndex,
                        a,
                        trueScore,
                        countBetter,
                        higherIsBetter,
                        chunkSize,
                        numEntitiesInt,
                    )
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

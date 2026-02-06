package jp.live.ugai.tugraph.eval

import ai.djl.inference.Predictor
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import jp.live.ugai.tugraph.TransR

class ResultEvalTransR(
    inputList: List<LongArray>,
    manager: NDManager,
    predictor: Predictor<NDList, NDList>,
    numEntities: Long,
    val transR: TransR,
    higherIsBetter: Boolean = false,
    filtered: Boolean = false,
) : ResultEval(inputList, manager, predictor, numEntities, higherIsBetter, filtered) {
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
        val entities = transR.getEntities()
        val edges = transR.getEdges()
        val matrix = transR.getMatrix()
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
                    val relIdsArr = relIds.toLongArray()
                    val indicesByRel = mutableMapOf<Long, MutableList<Int>>()
                    for (i in relIdsArr.indices) {
                        val relId = relIdsArr[i]
                        indicesByRel.getOrPut(relId) { mutableListOf() }.add(i)
                    }
                    val countBetterAll = LongArray(batchSize)
                    for ((relId, indices) in indicesByRel) {
                        val idxArr = indices.toIntArray()
                        batchManager.create(idxArr).use { idxNd ->
                            val headIdsNd =
                                if (mode == EvalMode.TAIL) {
                                    baseCol0.get(idxNd)
                                } else {
                                    trueIdsFlat.get(idxNd)
                                }
                            val tailIdsNd =
                                if (mode == EvalMode.TAIL) {
                                    trueIdsFlat.get(idxNd)
                                } else {
                                    baseCol1.get(idxNd)
                                }
                            batchManager.newSubManager().use { relManager ->
                                val hIds = headIdsNd.also { it.attach(relManager) }
                                val tIds = tailIdsNd.also { it.attach(relManager) }
                                val relIdNd = relManager.create(longArrayOf(relId))
                                val h = entities.get(hIds).also { it.attach(relManager) }
                                val t = entities.get(tIds).also { it.attach(relManager) }
                                val r = edges.get(relIdNd).also { it.attach(relManager) }
                                val matrixRel =
                                    matrix.get(relIdNd).reshape(r.shape[1], h.shape[1]).also { it.attach(relManager) }
                                val mT = matrixRel.transpose().also { it.attach(relManager) }
                                val hP = h.matMul(mT).also { it.attach(relManager) }
                                val tP = t.matMul(mT).also { it.attach(relManager) }
                                val base =
                                    if (mode == EvalMode.TAIL) {
                                        hP.add(r)
                                    } else {
                                        r.sub(tP)
                                    }.also { it.attach(relManager) }
                                val ts2d =
                                    if (mode == EvalMode.TAIL) {
                                        base.sub(tP).abs().sum(intArrayOf(1))
                                    } else {
                                        base.add(hP).abs().sum(intArrayOf(1))
                                    }.reshape(idxArr.size.toLong(), 1).also { it.attach(relManager) }

                                val countBetter =
                                    batchManager.zeros(
                                        Shape(idxArr.size.toLong()),
                                        DataType.INT64,
                                    )
                                var chunkStart = 0
                                while (chunkStart < numEntitiesInt) {
                                    val chunkEnd = minOf(chunkStart + chunkSize, numEntitiesInt)
                                    entities
                                        .get(NDIndex("$chunkStart:$chunkEnd, :"))
                                        .use { entityChunk ->
                                            relManager.newSubManager().use { chunkManager ->
                                                val projChunk = entityChunk.matMul(mT).also { it.attach(chunkManager) }
                                                val baseExp = base.expandDims(1).also { it.attach(chunkManager) }
                                                val projExp = projChunk.expandDims(0).also { it.attach(chunkManager) }
                                                val diff =
                                                    if (mode == EvalMode.TAIL) {
                                                        baseExp.sub(projExp)
                                                    } else {
                                                        baseExp.add(projExp)
                                                    }.also { it.attach(chunkManager) }
                                                val scores = diff.abs().sum(intArrayOf(2)).also { it.attach(chunkManager) }
                                                val countBetterNd =
                                                    if (higherIsBetter) {
                                                        scores.gt(ts2d).sum(intArrayOf(1))
                                                    } else {
                                                        scores.lt(ts2d).sum(intArrayOf(1))
                                                    }
                                                countBetter.addi(countBetterNd)
                                                countBetterNd.close()
                                            }
                                        }
                                    chunkStart = chunkEnd
                                }
                                val subsetCounts = countBetter.toLongArray()
                                countBetter.close()
                                for (i in idxArr.indices) {
                                    countBetterAll[idxArr[i]] = subsetCounts[i]
                                }
                            }
                        }
                    }
                    for (count in countBetterAll) {
                        ranks[outIndex++] = (count + 1L).toInt()
                    }
                }
            }
            start = end
        }
        return if (outIndex == totalSize) ranks else ranks.copyOf(outIndex)
    }
}

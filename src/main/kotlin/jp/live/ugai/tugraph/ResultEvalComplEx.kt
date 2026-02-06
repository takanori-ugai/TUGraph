package jp.live.ugai.tugraph

import ai.djl.inference.Predictor
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape

class ResultEvalComplEx(
    inputList: List<LongArray>,
    manager: NDManager,
    predictor: Predictor<NDList, NDList>,
    numEntities: Long,
    val complEx: ComplEx,
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
        return computeRanksComplEx(evalBatchSize, entityChunkSize, mode, buildBatch)
    }

    private fun computeRanksComplEx(
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
        val entities = complEx.getEntities()
        val edges = complEx.getEdges()
        val embDim2 = entities.shape[1].toInt()
        require(embDim2 % 2 == 0) { "ComplEx embeddings must have even dimension (found $embDim2)." }
        val halfDim = embDim2 / 2
        val realIndex = NDIndex(":, 0:$halfDim")
        val imagIndex = NDIndex(":, $halfDim:")
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
                    val rRe = rels.get(realIndex).also { it.attach(batchManager) }
                    val rIm = rels.get(imagIndex).also { it.attach(batchManager) }
                    val fRe = fixed.get(realIndex).also { it.attach(batchManager) }
                    val fIm = fixed.get(imagIndex).also { it.attach(batchManager) }
                    val (a, b) =
                        if (mode == EvalMode.TAIL) {
                            val aVal = fRe.mul(rRe).sub(fIm.mul(rIm)).also { it.attach(batchManager) }
                            val bVal = fRe.mul(rIm).add(fIm.mul(rRe)).also { it.attach(batchManager) }
                            aVal to bVal
                        } else {
                            val aVal = rRe.mul(fRe).add(rIm.mul(fIm)).also { it.attach(batchManager) }
                            val bVal = rRe.mul(fIm).sub(rIm.mul(fRe)).also { it.attach(batchManager) }
                            aVal to bVal
                        }
                    val trueEmb = entities.get(trueIdsFlat).also { it.attach(batchManager) }
                    val tRe = trueEmb.get(realIndex).also { it.attach(batchManager) }
                    val tIm = trueEmb.get(imagIndex).also { it.attach(batchManager) }
                    val trueScore =
                        a.mul(tRe).add(b.mul(tIm)).sum(intArrayOf(1)).reshape(batchSize.toLong(), 1)
                            .also { it.attach(batchManager) }
                    val countBetter = batchManager.zeros(Shape(batchSize.toLong()), DataType.INT64)
                    var chunkStart = 0
                    while (chunkStart < numEntitiesInt) {
                        val chunkEnd = minOf(chunkStart + chunkSize, numEntitiesInt)
                        entities.get(NDIndex("$chunkStart:$chunkEnd, :")).use { entityChunk ->
                            val eRe = entityChunk.get(realIndex)
                            val eIm = entityChunk.get(imagIndex)
                            eRe.use { er ->
                                eIm.use { ei ->
                                    er.transpose().use { erT ->
                                        ei.transpose().use { eiT ->
                                            a.matMul(erT).use { sc ->
                                                b.matMul(eiT).use { sc.addi(it) }
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

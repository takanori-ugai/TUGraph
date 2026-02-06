package jp.live.ugai.tugraph

import ai.djl.inference.Predictor
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape

class ResultEvalRotatE(
    inputList: List<LongArray>,
    manager: NDManager,
    predictor: Predictor<NDList, NDList>,
    numEntities: Long,
    val rotatE: RotatE,
    higherIsBetter: Boolean = false,
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
        return computeRanksRotatE(evalBatchSize, entityChunkSize, mode, buildBatch)
    }

    private fun computeRanksRotatE(
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
        val entities = rotatE.getEntities()
        val edges = rotatE.getEdges()
        val embDim2 = entities.shape[1].toInt()
        require(embDim2 % 2 == 0) { "RotatE embeddings must have even dimension (found $embDim2)." }
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
                    val rCos = rels.cos().also { it.attach(batchManager) }
                    val rSin = rels.sin().also { it.attach(batchManager) }
                    val fRe = fixed.get(realIndex).also { it.attach(batchManager) }
                    val fIm = fixed.get(imagIndex).also { it.attach(batchManager) }
                    val rotRe =
                        if (mode == EvalMode.TAIL) {
                            fRe.mul(rCos).sub(fIm.mul(rSin))
                        } else {
                            fRe.mul(rCos).add(fIm.mul(rSin))
                        }.also { it.attach(batchManager) }
                    val rotIm =
                        if (mode == EvalMode.TAIL) {
                            fRe.mul(rSin).add(fIm.mul(rCos))
                        } else {
                            fIm.mul(rCos).sub(fRe.mul(rSin))
                        }.also { it.attach(batchManager) }
                    val trueEmb = entities.get(trueIdsFlat).also { it.attach(batchManager) }
                    val tRe = trueEmb.get(realIndex).also { it.attach(batchManager) }
                    val tIm = trueEmb.get(imagIndex).also { it.attach(batchManager) }
                    val trueScore =
                        rotRe.sub(tRe).abs().add(rotIm.sub(tIm).abs()).sum(intArrayOf(1))
                            .reshape(batchSize.toLong(), 1)
                            .also { it.attach(batchManager) }
                    val countBetter = batchManager.zeros(Shape(batchSize.toLong()), DataType.INT64)
                    val rotReExp = rotRe.expandDims(1).also { it.attach(batchManager) }
                    val rotImExp = rotIm.expandDims(1).also { it.attach(batchManager) }
                    var chunkStart = 0
                    while (chunkStart < numEntitiesInt) {
                        val chunkEnd = minOf(chunkStart + chunkSize, numEntitiesInt)
                        entities.get(NDIndex("$chunkStart:$chunkEnd, :")).use { entityChunk ->
                            val eRe = entityChunk.get(realIndex)
                            val eIm = entityChunk.get(imagIndex)
                            eRe.use { re ->
                                eIm.use { im ->
                                    re.expandDims(0).use { reExp ->
                                        im.expandDims(0).use { imExp ->
                                            val diffRe = rotReExp.sub(reExp)
                                            val diffIm = rotImExp.sub(imExp)
                                            diffRe.use { dr ->
                                                diffIm.use { di ->
                                                    val scores = dr.abs().add(di.abs()).sum(intArrayOf(2))
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

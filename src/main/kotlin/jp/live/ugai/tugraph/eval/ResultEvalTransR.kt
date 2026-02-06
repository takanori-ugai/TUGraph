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
                            headIdsNd.use { hIds ->
                                tailIdsNd.use { tIds ->
                                    batchManager.create(longArrayOf(relId)).use { relIdNd ->
                                        val heads = entities.get(hIds)
                                        val tails = entities.get(tIds)
                                        val relEmb = edges.get(relIdNd)
                                        heads.use { h ->
                                            tails.use { t ->
                                                relEmb.use { r ->
                                                    val matrixRel = matrix.get(relIdNd).reshape(r.shape[1], h.shape[1])
                                                    matrixRel.use { mRel ->
                                                        val matrixT = mRel.transpose()
                                                        matrixT.use { mT ->
                                                            val headProj = h.matMul(mT)
                                                            val tailProj = t.matMul(mT)
                                                            headProj.use { hP ->
                                                                tailProj.use { tP ->
                                                                    val base =
                                                                        if (mode == EvalMode.TAIL) {
                                                                            hP.add(r)
                                                                        } else {
                                                                            r.sub(tP)
                                                                        }
                                                                    base.use { b ->
                                                                        val trueScore2d =
                                                                            if (mode == EvalMode.TAIL) {
                                                                                b.sub(tP).abs().sum(intArrayOf(1))
                                                                            } else {
                                                                                b.add(hP).abs().sum(intArrayOf(1))
                                                                            }.reshape(idxArr.size.toLong(), 1)
                                                                        trueScore2d.use { ts2d ->
                                                                            val countBetter =
                                                                                batchManager.zeros(
                                                                                    Shape(idxArr.size.toLong()),
                                                                                    DataType.INT64,
                                                                                )
                                                                            var chunkStart = 0
                                                                            while (chunkStart < numEntitiesInt) {
                                                                                val chunkEnd =
                                                                                    minOf(
                                                                                        chunkStart + chunkSize,
                                                                                        numEntitiesInt,
                                                                                    )
                                                                                entities
                                                                                    .get(
                                                                                        NDIndex("$chunkStart:$chunkEnd, :"),
                                                                                    ).use { entityChunk ->
                                                                                        val projChunk = entityChunk.matMul(mT)
                                                                                        projChunk.use { pChunk ->
                                                                                            val baseExp = b.expandDims(1)
                                                                                            val projExp = pChunk.expandDims(0)
                                                                                            baseExp.use { be ->
                                                                                                projExp.use { pe ->
                                                                                                    val diff =
                                                                                                        if (mode == EvalMode.TAIL) {
                                                                                                            be.sub(pe)
                                                                                                        } else {
                                                                                                            be.add(pe)
                                                                                                        }
                                                                                                    diff.use { d ->
                                                                                                        val scores =
                                                                                                            d.abs().sum(
                                                                                                                intArrayOf(2),
                                                                                                            )
                                                                                                        scores.use { sc ->
                                                                                                            val countBetterNd =
                                                                                                                if (higherIsBetter) {
                                                                                                                    sc
                                                                                                                        .gt(
                                                                                                                            ts2d,
                                                                                                                        ).sum(
                                                                                                                            intArrayOf(
                                                                                                                                1,
                                                                                                                            ),
                                                                                                                        )
                                                                                                                } else {
                                                                                                                    sc
                                                                                                                        .lt(
                                                                                                                            ts2d,
                                                                                                                        ).sum(
                                                                                                                            intArrayOf(
                                                                                                                                1,
                                                                                                                            ),
                                                                                                                        )
                                                                                                                }
                                                                                                            countBetter.addi(
                                                                                                                countBetterNd,
                                                                                                            )
                                                                                                            countBetterNd.close()
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                            }
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
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
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

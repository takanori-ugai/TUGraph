package jp.live.ugai.tugraph.eval

import ai.djl.inference.Predictor
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import jp.live.ugai.tugraph.*
import kotlin.math.sqrt

class ResultEvalHyperComplEx(
    inputList: List<LongArray>,
    manager: NDManager,
    predictor: Predictor<NDList, NDList>,
    numEntities: Long,
    val hyperComplEx: HyperComplEx,
    higherIsBetter: Boolean = true,
    filtered: Boolean = false,
) : ResultEval(inputList, manager, predictor, numEntities, higherIsBetter, filtered) {
    private val col0Index = NDIndex(":, 0")
    private val col1Index = NDIndex(":, 1")
    private val sumAxis2d = intArrayOf(1)
    private val sumAxis3d = intArrayOf(2)

    protected override fun computeRanks(
        evalBatchSize: Int,
        entityChunkSize: Int,
        mode: EvalMode,
        buildBatch: (start: Int, end: Int) -> EvalBatch,
    ): IntArray {
        val model = hyperComplEx
        val totalSize = inputList.size
        if (totalSize == 0) {
            return IntArray(0)
        }
        val ranks = IntArray(totalSize)
        var outIndex = 0
        val entitiesH = model.getEntitiesH()
        val entitiesC = model.getEntitiesC()
        val entitiesE = model.getEntitiesE()
        val edgesH = model.getEdgesH()
        val edgesC = model.getEdgesC()
        val edgesE = model.getEdgesE()
        val attn = model.getAttention()
        val dimC = model.dimC.toInt()
        val realIndex2d = NDIndex(":, 0:$dimC")
        val imagIndex2d = NDIndex(":, $dimC:")
        val chunkSize = maxOf(1, entityChunkSize)
        val numEntitiesInt = numEntities.toInt()
        var start = 0
        try {
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
                        val headIds = if (mode == EvalMode.TAIL) baseCol0 else trueIdsFlat
                        val tailIds = if (mode == EvalMode.TAIL) trueIdsFlat else baseCol1

                        val hH = entitiesH.get(headIds).also { it.attach(batchManager) }
                        val rH = edgesH.get(relIds).also { it.attach(batchManager) }
                        val tH = entitiesH.get(tailIds).also { it.attach(batchManager) }
                        val hC = entitiesC.get(headIds).also { it.attach(batchManager) }
                        val rC = edgesC.get(relIds).also { it.attach(batchManager) }
                        val tC = entitiesC.get(tailIds).also { it.attach(batchManager) }
                        val hE = entitiesE.get(headIds).also { it.attach(batchManager) }
                        val rE = edgesE.get(relIds).also { it.attach(batchManager) }
                        val tE = entitiesE.get(tailIds).also { it.attach(batchManager) }

                        val relAttn = attn.get(relIds).also { it.attach(batchManager) }
                        val alpha = relAttn.softmax(1).also { it.attach(batchManager) }
                        val alphaH = alpha.get(NDIndex(":, 0")).also { it.attach(batchManager) }
                        val alphaC = alpha.get(NDIndex(":, 1")).also { it.attach(batchManager) }
                        val alphaE = alpha.get(NDIndex(":, 2")).also { it.attach(batchManager) }

                        val mpH = maybeCast(hH, DataType.FLOAT16, model).also { it.attach(batchManager) }
                        val mpR = maybeCast(rH, DataType.FLOAT16, model).also { it.attach(batchManager) }
                        val mpT = maybeCast(tH, DataType.FLOAT16, model).also { it.attach(batchManager) }
                        val mpHC = maybeCast(hC, DataType.FLOAT16, model).also { it.attach(batchManager) }
                        val mpRC = maybeCast(rC, DataType.FLOAT16, model).also { it.attach(batchManager) }
                        val mpTC = maybeCast(tC, DataType.FLOAT16, model).also { it.attach(batchManager) }
                        val mpHE = maybeCast(hE, DataType.FLOAT16, model).also { it.attach(batchManager) }
                        val mpRE = maybeCast(rE, DataType.FLOAT16, model).also { it.attach(batchManager) }
                        val mpTE = maybeCast(tE, DataType.FLOAT16, model).also { it.attach(batchManager) }

                        val hyperTrue = hyperbolicScore2d(mpH, mpR, mpT, model).also { it.attach(batchManager) }
                        val complexTrue =
                            complexScore2d(mpHC, mpRC, mpTC, realIndex2d, imagIndex2d).also {
                                it.attach(batchManager)
                            }
                        val euclidTrue = euclideanScore2d(mpHE, mpRE, mpTE).also { it.attach(batchManager) }
                        val trueScore =
                            alphaH.mul(hyperTrue)
                                .add(alphaC.mul(complexTrue))
                                .add(alphaE.mul(euclidTrue))
                                .reshape(batchSize.toLong(), 1)
                                .also { it.attach(batchManager) }

                        val countBetter = batchManager.zeros(Shape(batchSize.toLong()), DataType.INT64)
                        val alphaHExp = alphaH.reshape(batchSize.toLong(), 1).also { it.attach(batchManager) }
                        val alphaCExp = alphaC.reshape(batchSize.toLong(), 1).also { it.attach(batchManager) }
                        val alphaEExp = alphaE.reshape(batchSize.toLong(), 1).also { it.attach(batchManager) }

                        var chunkStart = 0
                        while (chunkStart < numEntitiesInt) {
                            val chunkEnd = minOf(chunkStart + chunkSize, numEntitiesInt)
                            entitiesH.get(NDIndex("$chunkStart:$chunkEnd, :")).use { chunkH ->
                                entitiesC.get(NDIndex("$chunkStart:$chunkEnd, :")).use { chunkC ->
                                    entitiesE.get(NDIndex("$chunkStart:$chunkEnd, :")).use { chunkE ->
                                        val scores =
                                            if (mode == EvalMode.TAIL) {
                                                scoreTailChunk(
                                                    mpH,
                                                    mpR,
                                                    mpHC,
                                                    mpRC,
                                                    mpHE,
                                                    mpRE,
                                                    chunkH,
                                                    chunkC,
                                                    chunkE,
                                                    alphaHExp,
                                                    alphaCExp,
                                                    alphaEExp,
                                                    model,
                                                    realIndex2d,
                                                    imagIndex2d,
                                                )
                                            } else {
                                                scoreHeadChunk(
                                                    mpR,
                                                    mpT,
                                                    mpRC,
                                                    mpTC,
                                                    mpRE,
                                                    mpTE,
                                                    chunkH,
                                                    chunkC,
                                                    chunkE,
                                                    alphaHExp,
                                                    alphaCExp,
                                                    alphaEExp,
                                                    model,
                                                    realIndex2d,
                                                    imagIndex2d,
                                                )
                                            }
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
        } finally {
            entitiesH.close()
            entitiesC.close()
            entitiesE.close()
            edgesH.close()
            edgesC.close()
            edgesE.close()
        }
        return if (outIndex == totalSize) ranks else ranks.copyOf(outIndex)
    }

    private fun maybeCast(
        array: NDArray,
        target: DataType,
        model: HyperComplEx,
    ): NDArray {
        return if (model.mixedPrecision && array.dataType != target) {
            array.toType(target, false)
        } else {
            array
        }
    }

    private fun hyperbolicScore2d(
        h: NDArray,
        r: NDArray,
        t: NDArray,
        model: HyperComplEx,
    ): NDArray {
        val hPlusR = mobiusAdd2d(h, r, model.curvature)
        return hyperbolicDistance2d(hPlusR, t, model.curvature).neg()
    }

    private fun complexScore2d(
        h: NDArray,
        r: NDArray,
        t: NDArray,
        realIndex: NDIndex,
        imagIndex: NDIndex,
    ): NDArray {
        val hRe = h.get(realIndex)
        val hIm = h.get(imagIndex)
        val rRe = r.get(realIndex)
        val rIm = r.get(imagIndex)
        val tRe = t.get(realIndex)
        val tIm = t.get(imagIndex)
        val term1 = hRe.mul(rRe).sub(hIm.mul(rIm))
        val term2 = hRe.mul(rIm).add(hIm.mul(rRe))
        val sum1 = term1.mul(tRe)
        val sum2 = term2.mul(tIm)
        return sum1.add(sum2).sum(sumAxis2d)
    }

    private fun euclideanScore2d(
        h: NDArray,
        r: NDArray,
        t: NDArray,
    ): NDArray {
        return h.add(r).sub(t).pow(2).sum(sumAxis2d).neg()
    }

    private fun scoreTailChunk(
        hH: NDArray,
        rH: NDArray,
        hC: NDArray,
        rC: NDArray,
        hE: NDArray,
        rE: NDArray,
        chunkH: NDArray,
        chunkC: NDArray,
        chunkE: NDArray,
        alphaH: NDArray,
        alphaC: NDArray,
        alphaE: NDArray,
        model: HyperComplEx,
        realIndex2d: NDIndex,
        imagIndex2d: NDIndex,
    ): NDArray {
        val baseH = mobiusAdd2d(hH, rH, model.curvature)
        baseH.use { base ->
            val hyper =
                hyperbolicDistance3d(
                    base.expandDims(1),
                    chunkH.expandDims(0),
                    model.curvature,
                ).neg()
            hyper.use { hScore ->
                val complex = complexScoreTail(hC, rC, chunkC, realIndex2d, imagIndex2d)
                complex.use { cScore ->
                    val euclid = euclideanScoreTail(hE, rE, chunkE)
                    euclid.use { eScore ->
                        return alphaH.mul(hScore).add(alphaC.mul(cScore)).add(alphaE.mul(eScore))
                    }
                }
            }
        }
    }

    private fun scoreHeadChunk(
        rH: NDArray,
        tH: NDArray,
        rC: NDArray,
        tC: NDArray,
        rE: NDArray,
        tE: NDArray,
        chunkH: NDArray,
        chunkC: NDArray,
        chunkE: NDArray,
        alphaH: NDArray,
        alphaC: NDArray,
        alphaE: NDArray,
        model: HyperComplEx,
        realIndex2d: NDIndex,
        imagIndex2d: NDIndex,
    ): NDArray {
        val hyper =
            hyperbolicScoreHead(rH, tH, chunkH, model.curvature)
        hyper.use { hScore ->
            val complex = complexScoreHead(rC, tC, chunkC, realIndex2d, imagIndex2d)
            complex.use { cScore ->
                val euclid = euclideanScoreHead(rE, tE, chunkE)
                euclid.use { eScore ->
                    return alphaH.mul(hScore).add(alphaC.mul(cScore)).add(alphaE.mul(eScore))
                }
            }
        }
    }

    private fun complexScoreTail(
        h: NDArray,
        r: NDArray,
        tChunk: NDArray,
        realIndex: NDIndex,
        imagIndex: NDIndex,
    ): NDArray {
        val hRe = h.get(realIndex)
        val hIm = h.get(imagIndex)
        val rRe = r.get(realIndex)
        val rIm = r.get(imagIndex)
        val tRe = tChunk.get(realIndex)
        val tIm = tChunk.get(imagIndex)
        val term1 = hRe.mul(rRe).sub(hIm.mul(rIm))
        val term2 = hRe.mul(rIm).add(hIm.mul(rRe))
        return term1.matMul(tRe.transpose()).add(term2.matMul(tIm.transpose()))
    }

    private fun complexScoreHead(
        r: NDArray,
        t: NDArray,
        hChunk: NDArray,
        realIndex2d: NDIndex,
        imagIndex2d: NDIndex,
    ): NDArray {
        val hRe = hChunk.get(realIndex2d)
        val hIm = hChunk.get(imagIndex2d)
        val rRe = r.get(realIndex2d)
        val rIm = r.get(imagIndex2d)
        val tRe = t.get(realIndex2d)
        val tIm = t.get(imagIndex2d)
        val hReExp = hRe.expandDims(0)
        val hImExp = hIm.expandDims(0)
        val rReExp = rRe.expandDims(1)
        val rImExp = rIm.expandDims(1)
        val tReExp = tRe.expandDims(1)
        val tImExp = tIm.expandDims(1)
        val term1 = hReExp.mul(rReExp).sub(hImExp.mul(rImExp))
        val term2 = hReExp.mul(rImExp).add(hImExp.mul(rReExp))
        return term1.mul(tReExp).add(term2.mul(tImExp)).sum(sumAxis3d)
    }

    private fun euclideanScoreTail(
        h: NDArray,
        r: NDArray,
        tChunk: NDArray,
    ): NDArray {
        val base = h.add(r)
        return base.expandDims(1).sub(tChunk.expandDims(0)).pow(2).sum(sumAxis3d).neg()
    }

    private fun euclideanScoreHead(
        r: NDArray,
        t: NDArray,
        hChunk: NDArray,
    ): NDArray {
        val base = r.sub(t)
        return hChunk.expandDims(0).add(base.expandDims(1)).pow(2).sum(sumAxis3d).neg()
    }

    private fun hyperbolicScoreHead(
        r: NDArray,
        t: NDArray,
        hChunk: NDArray,
        curvature: Float,
    ): NDArray {
        val hExp = hChunk.expandDims(0)
        val rExp = r.expandDims(1)
        val tExp = t.expandDims(1)
        val hPlusR = mobiusAdd3d(hExp, rExp, curvature)
        return hyperbolicDistance3d(hPlusR, tExp, curvature).neg()
    }

    private fun mobiusAdd2d(
        x: NDArray,
        y: NDArray,
        curvature: Float,
    ): NDArray {
        val xy = x.mul(y).sum(sumAxis2d).reshape(x.shape[0], 1)
        val x2 = x.pow(2).sum(sumAxis2d).reshape(x.shape[0], 1)
        val y2 = y.pow(2).sum(sumAxis2d).reshape(x.shape[0], 1)
        val twoC = 2.0f * curvature
        val numLeft = x.mul(xy.mul(twoC).add(y2.mul(curvature)).add(1f))
        val numRight = y.mul(x2.mul(-curvature).add(1f))
        val numerator = numLeft.add(numRight)
        val denominator =
            xy.mul(twoC).add(x2.mul(y2).mul(curvature * curvature)).add(1f)
        return numerator.div(denominator)
    }

    private fun mobiusAdd3d(
        x: NDArray,
        y: NDArray,
        curvature: Float,
    ): NDArray {
        val xy = x.mul(y).sum(sumAxis3d).reshape(x.shape[0], x.shape[1], 1)
        val x2 = x.pow(2).sum(sumAxis3d).reshape(x.shape[0], x.shape[1], 1)
        val y2 = y.pow(2).sum(sumAxis3d).reshape(y.shape[0], y.shape[1], 1)
        val twoC = 2.0f * curvature
        val numLeft = x.mul(xy.mul(twoC).add(y2.mul(curvature)).add(1f))
        val numRight = y.mul(x2.mul(-curvature).add(1f))
        val numerator = numLeft.add(numRight)
        val denominator =
            xy.mul(twoC).add(x2.mul(y2).mul(curvature * curvature)).add(1f)
        return numerator.div(denominator)
    }

    private fun hyperbolicDistance2d(
        x: NDArray,
        y: NDArray,
        curvature: Float,
    ): NDArray {
        val diff2 = x.sub(y).pow(2).sum(sumAxis2d)
        val x2 = x.pow(2).sum(sumAxis2d)
        val y2 = y.pow(2).sum(sumAxis2d)
        val denom = x2.mul(-curvature).add(1f).mul(y2.mul(-curvature).add(1f))
        val term = diff2.mul(2f).div(denom).add(1f)
        val termClamped = term.maximum(1.0f + 1.0e-6f)
        val acosh = acosh(termClamped)
        return acosh.div(sqrt(curvature.toDouble()).toFloat())
    }

    private fun hyperbolicDistance3d(
        x: NDArray,
        y: NDArray,
        curvature: Float,
    ): NDArray {
        val diff2 = x.sub(y).pow(2).sum(sumAxis3d)
        val x2 = x.pow(2).sum(sumAxis3d)
        val y2 = y.pow(2).sum(sumAxis3d)
        val denom = x2.mul(-curvature).add(1f).mul(y2.mul(-curvature).add(1f))
        val term = diff2.mul(2f).div(denom).add(1f)
        val termClamped = term.maximum(1.0f + 1.0e-6f)
        val acosh = acosh(termClamped)
        return acosh.div(sqrt(curvature.toDouble()).toFloat())
    }

    private fun acosh(x: NDArray): NDArray {
        val zMinus1 = x.sub(1f)
        val zPlus1 = x.add(1f)
        val sqrtProd = zMinus1.mul(zPlus1).sqrt()
        return x.add(sqrtProd).log()
    }
}

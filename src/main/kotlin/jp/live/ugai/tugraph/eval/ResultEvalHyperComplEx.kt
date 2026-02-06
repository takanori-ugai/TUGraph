package jp.live.ugai.tugraph.eval

import ai.djl.inference.Predictor
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import jp.live.ugai.tugraph.HyperComplEx
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
                            alphaH
                                .mul(hyperTrue)
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
            attn.close()
        }
        return if (outIndex == totalSize) ranks else ranks.copyOf(outIndex)
    }

    private fun maybeCast(
        array: NDArray,
        target: DataType,
        model: HyperComplEx,
    ): NDArray =
        if (model.mixedPrecision && array.dataType != target) {
            array.toType(target, false)
        } else {
            array
        }

    private fun hyperbolicScore2d(
        h: NDArray,
        r: NDArray,
        t: NDArray,
        model: HyperComplEx,
    ): NDArray {
        val parent = h.manager
        return parent.newSubManager().use { sm ->
            val hPlusR = mobiusAdd2d(h, r, model.curvature).also { it.attach(sm) }
            val dist = hyperbolicDistance2d(hPlusR, t, model.curvature).also { it.attach(sm) }
            dist.neg().also { it.attach(parent) }
        }
    }

    private fun complexScore2d(
        h: NDArray,
        r: NDArray,
        t: NDArray,
        realIndex: NDIndex,
        imagIndex: NDIndex,
    ): NDArray {
        val parent = h.manager
        return parent.newSubManager().use { sm ->
            val hRe = h.get(realIndex).also { it.attach(sm) }
            val hIm = h.get(imagIndex).also { it.attach(sm) }
            val rRe = r.get(realIndex).also { it.attach(sm) }
            val rIm = r.get(imagIndex).also { it.attach(sm) }
            val tRe = t.get(realIndex).also { it.attach(sm) }
            val tIm = t.get(imagIndex).also { it.attach(sm) }
            val term1 = hRe.mul(rRe).sub(hIm.mul(rIm)).also { it.attach(sm) }
            val term2 = hRe.mul(rIm).add(hIm.mul(rRe)).also { it.attach(sm) }
            val sum1 = term1.mul(tRe).also { it.attach(sm) }
            val sum2 = term2.mul(tIm).also { it.attach(sm) }
            sum1.add(sum2).sum(sumAxis2d).also { it.attach(parent) }
        }
    }

    private fun euclideanScore2d(
        h: NDArray,
        r: NDArray,
        t: NDArray,
    ): NDArray =
        h.manager.newSubManager().use { sm ->
            val parent = h.manager
            val base = h.add(r).also { it.attach(sm) }
            base
                .sub(t)
                .pow(2)
                .sum(sumAxis2d)
                .neg()
                .also { it.attach(parent) }
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
        val parent = hH.manager
        return parent.newSubManager().use { sm ->
            val baseH = mobiusAdd2d(hH, rH, model.curvature).also { it.attach(sm) }
            val baseExp = baseH.expandDims(1).also { it.attach(sm) }
            val chunkHExp = chunkH.expandDims(0).also { it.attach(sm) }
            val hyper = hyperbolicDistance3d(baseExp, chunkHExp, model.curvature).neg().also { it.attach(sm) }
            val complex = complexScoreTail(hC, rC, chunkC, realIndex2d, imagIndex2d).also { it.attach(sm) }
            val euclid = euclideanScoreTail(hE, rE, chunkE).also { it.attach(sm) }
            alphaH
                .mul(hyper)
                .add(alphaC.mul(complex))
                .add(alphaE.mul(euclid))
                .also { it.attach(parent) }
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
        val parent = rH.manager
        return parent.newSubManager().use { sm ->
            val hyper = hyperbolicScoreHead(rH, tH, chunkH, model.curvature).also { it.attach(sm) }
            val complex = complexScoreHead(rC, tC, chunkC, realIndex2d, imagIndex2d).also { it.attach(sm) }
            val euclid = euclideanScoreHead(rE, tE, chunkE).also { it.attach(sm) }
            alphaH
                .mul(hyper)
                .add(alphaC.mul(complex))
                .add(alphaE.mul(euclid))
                .also { it.attach(parent) }
        }
    }

    private fun complexScoreTail(
        h: NDArray,
        r: NDArray,
        tChunk: NDArray,
        realIndex: NDIndex,
        imagIndex: NDIndex,
    ): NDArray {
        val parent = h.manager
        return parent.newSubManager().use { sm ->
            val hRe = h.get(realIndex).also { it.attach(sm) }
            val hIm = h.get(imagIndex).also { it.attach(sm) }
            val rRe = r.get(realIndex).also { it.attach(sm) }
            val rIm = r.get(imagIndex).also { it.attach(sm) }
            val tRe = tChunk.get(realIndex).also { it.attach(sm) }
            val tIm = tChunk.get(imagIndex).also { it.attach(sm) }
            val term1 = hRe.mul(rRe).sub(hIm.mul(rIm)).also { it.attach(sm) }
            val term2 = hRe.mul(rIm).add(hIm.mul(rRe)).also { it.attach(sm) }
            val tReT = tRe.transpose().also { it.attach(sm) }
            val tImT = tIm.transpose().also { it.attach(sm) }
            term1.matMul(tReT).add(term2.matMul(tImT)).also { it.attach(parent) }
        }
    }

    private fun complexScoreHead(
        r: NDArray,
        t: NDArray,
        hChunk: NDArray,
        realIndex2d: NDIndex,
        imagIndex2d: NDIndex,
    ): NDArray {
        val parent = r.manager
        return parent.newSubManager().use { sm ->
            val hRe = hChunk.get(realIndex2d).also { it.attach(sm) }
            val hIm = hChunk.get(imagIndex2d).also { it.attach(sm) }
            val rRe = r.get(realIndex2d).also { it.attach(sm) }
            val rIm = r.get(imagIndex2d).also { it.attach(sm) }
            val tRe = t.get(realIndex2d).also { it.attach(sm) }
            val tIm = t.get(imagIndex2d).also { it.attach(sm) }
            val hReExp = hRe.expandDims(0).also { it.attach(sm) }
            val hImExp = hIm.expandDims(0).also { it.attach(sm) }
            val rReExp = rRe.expandDims(1).also { it.attach(sm) }
            val rImExp = rIm.expandDims(1).also { it.attach(sm) }
            val tReExp = tRe.expandDims(1).also { it.attach(sm) }
            val tImExp = tIm.expandDims(1).also { it.attach(sm) }
            val term1 = hReExp.mul(rReExp).sub(hImExp.mul(rImExp)).also { it.attach(sm) }
            val term2 = hReExp.mul(rImExp).add(hImExp.mul(rReExp)).also { it.attach(sm) }
            term1
                .mul(tReExp)
                .add(term2.mul(tImExp))
                .sum(sumAxis3d)
                .also { it.attach(parent) }
        }
    }

    private fun euclideanScoreTail(
        h: NDArray,
        r: NDArray,
        tChunk: NDArray,
    ): NDArray {
        val parent = h.manager
        return parent.newSubManager().use { sm ->
            val base = h.add(r).also { it.attach(sm) }
            val baseExp = base.expandDims(1).also { it.attach(sm) }
            val tExp = tChunk.expandDims(0).also { it.attach(sm) }
            baseExp
                .sub(tExp)
                .pow(2)
                .sum(sumAxis3d)
                .neg()
                .also { it.attach(parent) }
        }
    }

    private fun euclideanScoreHead(
        r: NDArray,
        t: NDArray,
        hChunk: NDArray,
    ): NDArray {
        val parent = r.manager
        return parent.newSubManager().use { sm ->
            val base = r.sub(t).also { it.attach(sm) }
            val hExp = hChunk.expandDims(0).also { it.attach(sm) }
            val baseExp = base.expandDims(1).also { it.attach(sm) }
            hExp
                .add(baseExp)
                .pow(2)
                .sum(sumAxis3d)
                .neg()
                .also { it.attach(parent) }
        }
    }

    private fun hyperbolicScoreHead(
        r: NDArray,
        t: NDArray,
        hChunk: NDArray,
        curvature: Float,
    ): NDArray {
        val parent = r.manager
        return parent.newSubManager().use { sm ->
            val hExp = hChunk.expandDims(0).also { it.attach(sm) }
            val rExp = r.expandDims(1).also { it.attach(sm) }
            val tExp = t.expandDims(1).also { it.attach(sm) }
            val hPlusR = mobiusAdd3d(hExp, rExp, curvature).also { it.attach(sm) }
            hyperbolicDistance3d(hPlusR, tExp, curvature).neg().also { it.attach(parent) }
        }
    }

    private fun mobiusAdd2d(
        x: NDArray,
        y: NDArray,
        curvature: Float,
    ): NDArray {
        val parent = x.manager
        return parent.newSubManager().use { sm ->
            val xy =
                x
                    .mul(y)
                    .sum(sumAxis2d)
                    .reshape(x.shape[0], 1)
                    .also { it.attach(sm) }
            val x2 =
                x
                    .pow(2)
                    .sum(sumAxis2d)
                    .reshape(x.shape[0], 1)
                    .also { it.attach(sm) }
            val y2 =
                y
                    .pow(2)
                    .sum(sumAxis2d)
                    .reshape(x.shape[0], 1)
                    .also { it.attach(sm) }
            val twoC = 2.0f * curvature
            val numLeft = x.mul(xy.mul(twoC).add(y2.mul(curvature)).add(1f)).also { it.attach(sm) }
            val numRight = y.mul(x2.mul(-curvature).add(1f)).also { it.attach(sm) }
            val numerator = numLeft.add(numRight).also { it.attach(sm) }
            val denominator =
                xy
                    .mul(twoC)
                    .add(x2.mul(y2).mul(curvature * curvature))
                    .add(1f)
                    .also { it.attach(sm) }
            numerator.div(denominator).also { it.attach(parent) }
        }
    }

    private fun mobiusAdd3d(
        x: NDArray,
        y: NDArray,
        curvature: Float,
    ): NDArray {
        val parent = x.manager
        return parent.newSubManager().use { sm ->
            val xy =
                x
                    .mul(y)
                    .sum(sumAxis3d)
                    .reshape(x.shape[0], x.shape[1], 1)
                    .also { it.attach(sm) }
            val x2 =
                x
                    .pow(2)
                    .sum(sumAxis3d)
                    .reshape(x.shape[0], x.shape[1], 1)
                    .also { it.attach(sm) }
            val y2 =
                y
                    .pow(2)
                    .sum(sumAxis3d)
                    .reshape(y.shape[0], y.shape[1], 1)
                    .also { it.attach(sm) }
            val twoC = 2.0f * curvature
            val numLeft = x.mul(xy.mul(twoC).add(y2.mul(curvature)).add(1f)).also { it.attach(sm) }
            val numRight = y.mul(x2.mul(-curvature).add(1f)).also { it.attach(sm) }
            val numerator = numLeft.add(numRight).also { it.attach(sm) }
            val denominator =
                xy
                    .mul(twoC)
                    .add(x2.mul(y2).mul(curvature * curvature))
                    .add(1f)
                    .also { it.attach(sm) }
            numerator.div(denominator).also { it.attach(parent) }
        }
    }

    private fun hyperbolicDistance2d(
        x: NDArray,
        y: NDArray,
        curvature: Float,
    ): NDArray {
        val parent = x.manager
        return parent.newSubManager().use { sm ->
            val diff2 =
                x
                    .sub(y)
                    .pow(2)
                    .sum(sumAxis2d)
                    .also { it.attach(sm) }
            val x2 = x.pow(2).sum(sumAxis2d).also { it.attach(sm) }
            val y2 = y.pow(2).sum(sumAxis2d).also { it.attach(sm) }
            val denom =
                x2
                    .mul(-curvature)
                    .add(1f)
                    .mul(y2.mul(-curvature).add(1f))
                    .also { it.attach(sm) }
            val term =
                diff2
                    .mul(2f)
                    .div(denom)
                    .add(1f)
                    .also { it.attach(sm) }
            val termClamped = term.maximum(1.0f + 1.0e-6f).also { it.attach(sm) }
            val acosh = acosh(termClamped).also { it.attach(sm) }
            acosh.div(sqrt(curvature.toDouble()).toFloat()).also { it.attach(parent) }
        }
    }

    private fun hyperbolicDistance3d(
        x: NDArray,
        y: NDArray,
        curvature: Float,
    ): NDArray {
        val parent = x.manager
        return parent.newSubManager().use { sm ->
            val diff2 =
                x
                    .sub(y)
                    .pow(2)
                    .sum(sumAxis3d)
                    .also { it.attach(sm) }
            val x2 = x.pow(2).sum(sumAxis3d).also { it.attach(sm) }
            val y2 = y.pow(2).sum(sumAxis3d).also { it.attach(sm) }
            val denom =
                x2
                    .mul(-curvature)
                    .add(1f)
                    .mul(y2.mul(-curvature).add(1f))
                    .also { it.attach(sm) }
            val term =
                diff2
                    .mul(2f)
                    .div(denom)
                    .add(1f)
                    .also { it.attach(sm) }
            val termClamped = term.maximum(1.0f + 1.0e-6f).also { it.attach(sm) }
            val acosh = acosh(termClamped).also { it.attach(sm) }
            acosh.div(sqrt(curvature.toDouble()).toFloat()).also { it.attach(parent) }
        }
    }

    private fun acosh(x: NDArray): NDArray {
        val parent = x.manager
        return parent.newSubManager().use { sm ->
            val zMinus1 = x.sub(1f).also { it.attach(sm) }
            val zPlus1 = x.add(1f).also { it.attach(sm) }
            val sqrtProd = zMinus1.mul(zPlus1).sqrt().also { it.attach(sm) }
            x.add(sqrtProd).log().also { it.attach(parent) }
        }
    }
}

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

    /**
     * Compute 1-based ranks for each evaluation triple by scoring all candidate entities and comparing
     * them to the ground-truth entity according to the HyperComplEx model.
     *
     * This method processes inputs in batches and in entity chunks, combines hyperbolic, complex,
     * and Euclidean score components with per-relation attention weights, and counts how many
     * candidate entities produce a better score than the true entity to produce a rank = count + 1.
     *
     * @param evalBatchSize Number of evaluation examples processed per batch.
     * @param entityChunkSize Maximum number of entities evaluated at once when scanning candidates.
     * @param mode Whether to replace HEAD or TAIL when scoring candidates.
     * @param buildBatch Function that builds an EvalBatch for a given [start, end) range of the
     *   input list; the returned EvalBatch must provide the base pair, true entity column, and
     *   batch size used for scoring.
     * @return An IntArray of 1-based ranks for each input triple (length equals number of inputs;
     *   may be shorter if processing was interrupted).
     */
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

    /**
     * Casts the given NDArray to the specified data type when the model is using mixed precision.
     *
     * @param array The array to potentially cast.
     * @param target The desired data type.
     * @param model The HyperComplEx model whose `mixedPrecision` flag controls casting.
     * @return The original array, or a converted array with `target` data type if `model.mixedPrecision` is true
     *   and the array's data type differs.
     */
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

    /**
     * Computes the negated hyperbolic distance between the Möbius sum of `h` and `r` and `t`.
     *
     * Uses the model's curvature to perform Möbius addition on `h` and `r`, then measures the
     * hyperbolic distance to `t` and returns its negation.
     *
     * @param h Head/entity embeddings in 2D representation.
     * @param r Relation embeddings in 2D representation.
     * @param t Tail/entity embeddings in 2D representation.
     * @param model HyperComplEx model providing curvature and related settings.
     * @return An NDArray of negated hyperbolic distances for each example.
     */
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

    /**
     * Computes the complex-component score for (h, r, t) by combining their real and imaginary parts
     * and summing across the feature axis.
     *
     * @param h Entity embeddings for the source (head or tail) in interleaved or split complex layout.
     * @param r Relation embeddings in the same layout as `h`.
     * @param t Candidate target entity embeddings aligned with `h`/`r`.
     * @param realIndex Index used to extract the real-part slice from complex-formatted arrays.
     * @param imagIndex Index used to extract the imaginary-part slice from complex-formatted arrays.
     * @return An NDArray containing the per-example complex scores summed over the feature axis.
     */
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

    /**
     * Computes the Euclidean score component for 2D embeddings: negative summed squared error of (h + r - t).
     *
     * @param h 2D embedding for the head (batch × features), with the feature axis at index 1.
     * @param r 2D embedding for the relation (batch × features), with the feature axis at index 1.
     * @param t 2D embedding for the tail (batch × features), with the feature axis at index 1.
     * @return 1D `NDArray` of per-example scores equal to -sum((h + r - t)^2) summed over the feature axis.
     */
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

    /**
     * Computes combined scores for candidate tail entities in a chunk by weighting
     * hyperbolic, complex, and Euclidean components with the provided attention alphas.
     *
     * @param hH Head embeddings in the hyperbolic space.
     * @param rH Relation embeddings in the hyperbolic space.
     * @param hC Head embeddings in the complex space.
     * @param rC Relation embeddings in the complex space.
     * @param hE Head embeddings in the Euclidean space.
     * @param rE Relation embeddings in the Euclidean space.
     * @param chunkH Candidate tail embeddings in the hyperbolic space (chunk).
     * @param chunkC Candidate tail embeddings in the complex space (chunk).
     * @param chunkE Candidate tail embeddings in the Euclidean space (chunk).
     * @param alphaH Attention weights for the hyperbolic component (broadcastable to result shape).
     * @param alphaC Attention weights for the complex component (broadcastable to result shape).
     * @param alphaE Attention weights for the Euclidean component (broadcastable to result shape).
     * @param model The HyperComplEx model providing curvature and mixed-precision settings.
     * @param realIndex2d Index for selecting real parts from 2D complex embeddings.
     * @param imagIndex2d Index for selecting imaginary parts from 2D complex embeddings.
     * @return An NDArray of combined scores for each input example against each candidate tail in the chunk.
     */
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

    /**
     * Compute combined scores for a batch of head candidates by aggregating hyperbolic,
     * complex, and Euclidean components weighted by the provided attention alphas.
     *
     * @param rH Relation embeddings in the hyperbolic space.
     * @param tH Tail embeddings in the hyperbolic space.
     * @param rC Relation embeddings in the complex space.
     * @param tC Tail embeddings in the complex space.
     * @param rE Relation embeddings in the Euclidean space.
     * @param tE Tail embeddings in the Euclidean space.
     * @param chunkH Chunk of candidate head embeddings in the hyperbolic space.
     * @param chunkC Chunk of candidate head embeddings in the complex space.
     * @param chunkE Chunk of candidate head embeddings in the Euclidean space.
     * @param alphaH Attention weights for the hyperbolic component (broadcastable to scores).
     * @param alphaC Attention weights for the complex component (broadcastable to scores).
     * @param alphaE Attention weights for the Euclidean component (broadcastable to scores).
     * @param model HyperComplEx model instance (used for hyperbolic curvature).
     * @param realIndex2d Index selector for real parts in 2D complex tensors.
     * @param imagIndex2d Index selector for imaginary parts in 2D complex tensors.
     * @return An NDArray containing the combined scores for each example in the batch against the candidate head chunk.
     */
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

    /**
     * Compute the complex-component scores for a batch of tail candidates given head and relation embeddings.
     *
     * Extracts real and imaginary parts using the provided indices and returns the complex score matrix
     * between each head-relation pair and each tail in the chunk.
     *
     * @param h Head embeddings containing interleaved real/imaginary parts.
     * @param r Relation embeddings containing interleaved real/imaginary parts.
     * @param tChunk Chunk of tail embeddings to score against.
     * @param realIndex NDIndex selecting the real components from embeddings.
     * @param imagIndex NDIndex selecting the imaginary components from embeddings.
     * @return An NDArray of scores with shape (batchSize, chunkSize) representing the complex component
     *         of the score for each head-relation pair against each tail candidate.
     */
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

    /**
     * Compute the Complex-space scoring component for candidate head entities given relations and tails.
     *
     * @param r Relation embeddings arranged for 2D access (contains real and imaginary parts).
     * @param t Tail embeddings arranged for 2D access (contains real and imaginary parts).
     * @param hChunk Chunk of candidate head embeddings to score against `r` and `t`.
     * @param realIndex2d NDIndex used to select the real components from 2D embedding arrays.
     * @param imagIndex2d NDIndex used to select the imaginary components from 2D embedding arrays.
     * @return An NDArray of shape (batchSize, chunkSize) containing the summed complex scores for each candidate head.
     */
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

    /**
     * Compute Euclidean-based scores for candidate tails given head and relation embeddings.
     *
     * Computes -sum((h + r - t)^2) over the embedding dimension for each pair of batch heads and candidate tails.
     *
     * @param h Head embeddings with shape (batchSize, dim).
     * @param r Relation embeddings with shape (batchSize, dim).
     * @param tChunk Candidate tail embeddings with shape (chunkSize, dim).
     * @return An NDArray of shape (batchSize, chunkSize) containing the negative squared Euclidean distances
     *   (higher values indicate closer/more similar candidates).
     */
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

    /**
     * Compute Euclidean scores for candidate head entities given relation and tail embeddings.
     *
     * @param r Relation embeddings with shape (batchSize, dim).
     * @param t Tail embeddings with shape (batchSize, dim).
     * @param hChunk Candidate head embeddings chunk with shape (chunkSize, dim).
     * @return An NDArray of shape (batchSize, chunkSize) containing the negative sum of squared
     *         differences (-(h + (r - t))^2 summed over the feature axis) for each candidate head.
     */
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

    /**
     * Compute hyperbolic scores for a chunk of candidate heads given a relation and target tail.
     *
     * @param r Relation embeddings in 2D form.
     * @param t Tail embeddings in 2D form.
     * @param hChunk Chunk of candidate head embeddings (may contain multiple candidates).
     * @param curvature Positive curvature scalar used for the hyperbolic geometry.
     * @return An `NDArray` of negated hyperbolic distances between (hChunk ⊕ r) and `t` under the given curvature;
     *   larger values indicate closer distance in hyperbolic space.
     */
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

    /**
     * Compute the Möbius addition of two batches of vectors under a given curvature.
     *
     * Both `x` and `y` are expected to be 2D NDArrays shaped (batch, features); the operation is applied per-row.
     *
     * @param x Left operand batch of vectors.
     * @param y Right operand batch of vectors.
     * @param curvature Positive float controlling the Möbius curvature.
     * @return An NDArray with the same shape as `x` and `y` containing the Möbius sum of each pair of rows.
     */
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

    /**
     * Compute the Möbius addition of two 3D tensors under the specified curvature.
     *
     * Performs Möbius addition elementwise across the last dimension for two tensors with matching shapes.
     *
     * @param x A 3D tensor where the last axis is the feature dimension used in the Möbius operation.
     * @param y A 3D tensor with the same shape as `x`.
     * @param curvature Positive curvature scalar controlling the hyperbolic geometry.
     * @return An NDArray with the same shape as `x` and `y` representing their Möbius sum.
     */
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

    /**
     * Compute the hyperbolic distance between corresponding 2D point tensors under the specified curvature.
     *
     * @param x Tensor of 2D points; must have the same shape as `y`.
     * @param y Tensor of 2D points; must have the same shape as `x`.
     * @param curvature Positive curvature scalar used for the hyperbolic metric.
     * @return An NDArray containing the hyperbolic distances between `x` and `y`.
     */
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

    /**
     * Compute the hyperbolic distance between corresponding vectors in `x` and `y` under the given curvature.
     *
     * @param x Tensor of points (batch-compatible) where the distance is measured from; last feature axis is folded into the distance.
     * @param y Tensor of points (batch-compatible) where the distance is measured to; must be broadcastable with `x`.
     * @param curvature Positive curvature scalar for the hyperbolic model.
     * @return An NDArray of distances computed per pair in `x` and `y`, with the feature axis reduced
     *   (shape reflects the remaining batch dimensions).
     */
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

    /**
     * Compute the elementwise inverse hyperbolic cosine (arcosh) of the input NDArray.
     *
     * @param x Input array; each element is expected to be greater than or equal to 1.
     * @return An NDArray containing the elementwise inverse hyperbolic cosine of `x`. Elements with `x < 1` will produce `NaN`.
     */
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

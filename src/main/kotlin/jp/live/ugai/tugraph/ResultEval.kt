package jp.live.ugai.tugraph

import ai.djl.inference.Predictor
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape

/**
 * A class for evaluating the results of a model.
 *
 * @property inputList The list of input triples.
 * @property manager The NDManager instance used for creating NDArrays.
 * @property predictor The Predictor instance for making predictions.
 * @property numEntities The number of entities in the data.
 */
class ResultEval(
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
    /** Optional TransE block for embedding-based evaluation. */
    val transE: TransE? = null,
    /** Optional TransR block for embedding-based evaluation. */
    val transR: TransR? = null,
    /** Optional DistMult block for embedding-based evaluation. */
    val distMult: DistMult? = null,
    /** Optional RotatE block for embedding-based evaluation. */
    val rotatE: RotatE? = null,
    /** Optional QuatE block for embedding-based evaluation. */
    val quatE: QuatE? = null,
    /** Optional ComplEx block for embedding-based evaluation. */
    val complEx: ComplEx? = null,
) {
    private val col0Index = NDIndex(":, 0")
    private val col1Index = NDIndex(":, 1")

    private data class EvalBatch(
        val basePair: NDArray,
        val trueIds: IntArray,
        val trueEntityCol: NDArray,
        val batchSize: Int,
        private val closeables: List<NDArray>,
    ) : AutoCloseable {
        override fun close() {
            closeables.forEach { it.close() }
        }
    }

    private data class ChunkInput(
        val modelInput: NDArray,
        private val closeables: List<NDArray>,
    ) : AutoCloseable {
        override fun close() {
            closeables.forEach { it.close() }
        }
    }

    /**
     * Compute per-example ranks under the TransE scoring model for either head or tail prediction.
     *
     * @param evalBatchSize Number of input triples processed per outer evaluation batch.
     * @param entityChunkSize Maximum number of candidate entities evaluated per inner chunk.
     * @param mode Whether to score candidates for the tail (EvalMode.TAIL) or the head (EvalMode.HEAD).
     * @param buildBatch Callback that builds an EvalBatch for inputs in the half-open range [start, end).
     * @return A list of 1-based ranks (one element per input) where each rank is the position of the true entity
     *         among all candidate entities according to TransE scores.
    private fun computeRanksTransE(
        evalBatchSize: Int,
        entityChunkSize: Int,
        mode: EvalMode,
        buildBatch: (start: Int, end: Int) -> EvalBatch,
    ): List<Int> {
        val transe = transE ?: return emptyList()
        val ranks = mutableListOf<Int>()
        val entities = transe.getEntities()
        val edges = transe.getEdges()
        val chunkSize = maxOf(1, entityChunkSize)
        var start = 0
        while (start < inputList.size) {
            val end = minOf(start + evalBatchSize, inputList.size)
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
                    val baseEmb =
                        when (mode) {
                            EvalMode.TAIL -> {
                                val heads = entities.get(baseCol0).also { it.attach(batchManager) }
                                val rels = edges.get(baseCol1).also { it.attach(batchManager) }
                                heads.add(rels).also { it.attach(batchManager) }
                            }
                            EvalMode.HEAD -> {
                                val rels = edges.get(baseCol0).also { it.attach(batchManager) }
                                val tails = entities.get(baseCol1).also { it.attach(batchManager) }
                                tails.sub(rels).also { it.attach(batchManager) }
                            }
                        }
                    val trueEmb = entities.get(trueIdsFlat).also { it.attach(batchManager) }
                    val trueScore =
                        baseEmb.sub(trueEmb).abs().sum(intArrayOf(1)).reshape(batchSize.toLong(), 1)
                            .also { it.attach(batchManager) }
                    val countBetter = batchManager.zeros(Shape(batchSize.toLong()), DataType.INT64)
                    val baseEmbExp = baseEmb.expandDims(1).also { it.attach(batchManager) }
                    var chunkStart = 0
                    while (chunkStart < numEntities.toInt()) {
                        val chunkEnd = minOf(chunkStart + chunkSize, numEntities.toInt())
                        entities.get(NDIndex("$chunkStart:$chunkEnd, :")).use { entityChunk ->
                            val entityExp = entityChunk.expandDims(0)
                            entityExp.use { ent ->
                                val diff = baseEmbExp.sub(ent)
                                diff.use { d ->
                                    val scores = d.abs().sum(intArrayOf(2))
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
                        ranks.add(rank.toInt())
                    }
                }
            }
            start = end
        }
        return ranks
    }

    private enum class EvalMode {
        TAIL,
        HEAD,
    }

    /**
     * Compute 1-based ranks for each input triple using the DistMult scoring model.
     *
     * Processes inputList in outer batches and evaluates candidate entities in inner chunks;
     * for each triple it counts how many candidate entities score better than the true entity
     * and produces rank = countBetter + 1.
     *
     * @param evalBatchSize Number of triples evaluated per outer batch.
     * @param entityChunkSize Maximum number of entities evaluated per inner chunk.
     * @param mode Evaluation mode indicating whether to predict TAIL or HEAD.
     * @param buildBatch Callback that constructs an EvalBatch for the half-open range [start, end).
     * @return A list of 1-based ranks (one per input triple) where each rank equals the count
     *         of candidate entities with a strictly better score than the true entity plus one.
     */
    private fun computeRanksDistMult(
        evalBatchSize: Int,
        entityChunkSize: Int,
        mode: EvalMode,
        buildBatch: (start: Int, end: Int) -> EvalBatch,
    ): List<Int> {
        val model = distMult ?: return emptyList()
        val ranks = mutableListOf<Int>()
        val entities = model.getEntities()
        val edges = model.getEdges()
        val chunkSize = maxOf(1, entityChunkSize)
        var start = 0
        while (start < inputList.size) {
            val end = minOf(start + evalBatchSize, inputList.size)
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
                        baseMul.mul(trueEmb).sum(intArrayOf(1)).reshape(batchSize.toLong(), 1)
                            .also { it.attach(batchManager) }
                    val countBetter = batchManager.zeros(Shape(batchSize.toLong()), DataType.INT64)
                    var chunkStart = 0
                    while (chunkStart < numEntities.toInt()) {
                        val chunkEnd = minOf(chunkStart + chunkSize, numEntities.toInt())
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
                        ranks.add(rank.toInt())
                    }
                }
            }
            start = end
        }
        return ranks
    }

    /**
     * Compute 1-based ranks for each input triple using the RotatE embedding model.
     *
     * This evaluates inputs in outer batches of size [evalBatchSize] and processes candidate entities
     * in inner chunks of size [entityChunkSize]. For each input, it rotates the fixed entity by the
     * relation and counts how many candidate entities have a "better" score according to the
     * instance's `higherIsBetter` flag; the rank is `countBetter + 1`. If `rotatE` is null, returns
     * an empty list.
     *
     * @param evalBatchSize Number of input triples processed per outer batch.
     * @param entityChunkSize Number of candidate entities evaluated per inner chunk.
     * @param mode Whether to predict TAIL (fix head+rel) or HEAD (fix rel+tail).
     * @param buildBatch Callback that constructs an EvalBatch for inputs in the half-open range [start, end).
     * @return A list of 1-based ranks (one rank per input triple) where `1` indicates the best rank.
     * @throws IllegalArgumentException if the RotatE embedding dimension is not even.
     */
    private fun computeRanksRotatE(
        evalBatchSize: Int,
        entityChunkSize: Int,
        mode: EvalMode,
        buildBatch: (start: Int, end: Int) -> EvalBatch,
    ): List<Int> {
        val model = rotatE ?: return emptyList()
        val ranks = mutableListOf<Int>()
        val entities = model.getEntities()
        val edges = model.getEdges()
        val embDim2 = entities.shape[1].toInt()
        require(embDim2 % 2 == 0) { "RotatE embeddings must have even dimension (found $embDim2)." }
        val halfDim = embDim2 / 2
        val realIndex = NDIndex(":, 0:$halfDim")
        val imagIndex = NDIndex(":, $halfDim:")
        val chunkSize = maxOf(1, entityChunkSize)
        var start = 0
        while (start < inputList.size) {
            val end = minOf(start + evalBatchSize, inputList.size)
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
                    var chunkStart = 0
                    while (chunkStart < numEntities.toInt()) {
                        val chunkEnd = minOf(chunkStart + chunkSize, numEntities.toInt())
                        entities.get(NDIndex("$chunkStart:$chunkEnd, :")).use { entityChunk ->
                            val eRe = entityChunk.get(realIndex)
                            val eIm = entityChunk.get(imagIndex)
                            eRe.use { re ->
                                eIm.use { im ->
                                    val diffRe = rotRe.expandDims(1).sub(re.expandDims(0))
                                    val diffIm = rotIm.expandDims(1).sub(im.expandDims(0))
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
                        chunkStart = chunkEnd
                    }
                    val ranksNd = countBetter.add(1)
                    val batchRanks = ranksNd.toLongArray()
                    ranksNd.close()
                    for (rank in batchRanks) {
                        ranks.add(rank.toInt())
                    }
                }
            }
            start = end
        }
        return ranks
    }

    /**
     * Computes ranking positions for each input triple using the QuatE embedding model.
     *
     * For each item in the evaluation input list this builds a batch of base (head,relation) or
     * (relation,tail) pairs depending on [mode], scores all candidate entities in chunks, and
     * returns the 1-based rank of the true entity for each item (1 means best).
     *
     * @param evalBatchSize Number of input triples processed per outer batch.
     * @param entityChunkSize Number of entity embeddings processed per inner chunk.
     * @param mode Whether to predict TAIL or HEAD for each base pair.
     * @param buildBatch Callback that constructs an EvalBatch for inputs in the range [start, end).
     * @return A list of 1-based ranks (one rank per input triple). If the class property `quatE` is null,
     *         an empty list is returned.
     * @throws IllegalArgumentException If the QuatE entity embedding dimension is not divisible by 4.
     */
    private fun computeRanksQuatE(
        evalBatchSize: Int,
        entityChunkSize: Int,
        mode: EvalMode,
        buildBatch: (start: Int, end: Int) -> EvalBatch,
    ): List<Int> {
        val model = quatE ?: return emptyList()
        val ranks = mutableListOf<Int>()
        val entities = model.getEntities()
        val edges = model.getEdges()
        val embDim4 = entities.shape[1].toInt()
        require(embDim4 % 4 == 0) { "QuatE embeddings must have dimension divisible by 4 (found $embDim4)." }
        val quarterDim = embDim4 / 4
        val rIndex = NDIndex(":, 0:$quarterDim")
        val iIndex = NDIndex(":, $quarterDim:${quarterDim * 2}")
        val jIndex = NDIndex(":, ${quarterDim * 2}:${quarterDim * 3}")
        val kIndex = NDIndex(":, ${quarterDim * 3}:")
        val chunkSize = maxOf(1, entityChunkSize)
        var start = 0
        while (start < inputList.size) {
            val end = minOf(start + evalBatchSize, inputList.size)
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
                    val rR = rels.get(rIndex).also { it.attach(batchManager) }
                    val rI = rels.get(iIndex).also { it.attach(batchManager) }
                    val rJ = rels.get(jIndex).also { it.attach(batchManager) }
                    val rK = rels.get(kIndex).also { it.attach(batchManager) }
                    val fR = fixed.get(rIndex).also { it.attach(batchManager) }
                    val fI = fixed.get(iIndex).also { it.attach(batchManager) }
                    val fJ = fixed.get(jIndex).also { it.attach(batchManager) }
                    val fK = fixed.get(kIndex).also { it.attach(batchManager) }
                    val aR: NDArray
                    val aI: NDArray
                    val aJ: NDArray
                    val aK: NDArray
                    if (mode == EvalMode.TAIL) {
                        val hrR = fR.mul(rR).sub(fI.mul(rI)).sub(fJ.mul(rJ)).sub(fK.mul(rK))
                        val hrI = fR.mul(rI).add(fI.mul(rR)).add(fJ.mul(rK)).sub(fK.mul(rJ))
                        val hrJ = fR.mul(rJ).sub(fI.mul(rK)).add(fJ.mul(rR)).add(fK.mul(rI))
                        val hrK = fR.mul(rK).add(fI.mul(rJ)).sub(fJ.mul(rI)).add(fK.mul(rR))
                        aR = hrR.also { it.attach(batchManager) }
                        aI = hrI.also { it.attach(batchManager) }
                        aJ = hrJ.also { it.attach(batchManager) }
                        aK = hrK.also { it.attach(batchManager) }
                    } else {
                        val tEmb = entities.get(trueIdsFlat).also { it.attach(batchManager) }
                        val tR = tEmb.get(rIndex).also { it.attach(batchManager) }
                        val tI = tEmb.get(iIndex).also { it.attach(batchManager) }
                        val tJ = tEmb.get(jIndex).also { it.attach(batchManager) }
                        val tK = tEmb.get(kIndex).also { it.attach(batchManager) }
                        aR = rR.mul(tR).add(rI.mul(tI)).add(rJ.mul(tJ)).add(rK.mul(tK)).also {
                            it.attach(batchManager)
                        }
                        aI = rR.mul(tI).sub(rI.mul(tR)).add(rJ.mul(tK)).sub(rK.mul(tJ)).also {
                            it.attach(batchManager)
                        }
                        aJ = rR.mul(tJ).sub(rJ.mul(tR)).add(rK.mul(tI)).sub(rI.mul(tK)).also {
                            it.attach(batchManager)
                        }
                        aK = rR.mul(tK).sub(rK.mul(tR)).add(rI.mul(tJ)).sub(rJ.mul(tI)).also {
                            it.attach(batchManager)
                        }
                    }
                    val trueScore =
                        if (mode == EvalMode.TAIL) {
                            val trueEmb = entities.get(trueIdsFlat).also { it.attach(batchManager) }
                            val tR = trueEmb.get(rIndex).also { it.attach(batchManager) }
                            val tI = trueEmb.get(iIndex).also { it.attach(batchManager) }
                            val tJ = trueEmb.get(jIndex).also { it.attach(batchManager) }
                            val tK = trueEmb.get(kIndex).also { it.attach(batchManager) }
                            aR.mul(tR).add(aI.mul(tI)).add(aJ.mul(tJ)).add(aK.mul(tK)).sum(intArrayOf(1))
                                .reshape(batchSize.toLong(), 1)
                                .also { it.attach(batchManager) }
                        } else {
                            val trueEmb = entities.get(trueIdsFlat).also { it.attach(batchManager) }
                            val hR = trueEmb.get(rIndex).also { it.attach(batchManager) }
                            val hI = trueEmb.get(iIndex).also { it.attach(batchManager) }
                            val hJ = trueEmb.get(jIndex).also { it.attach(batchManager) }
                            val hK = trueEmb.get(kIndex).also { it.attach(batchManager) }
                            aR.mul(hR).add(aI.mul(hI)).add(aJ.mul(hJ)).add(aK.mul(hK)).sum(intArrayOf(1))
                                .reshape(batchSize.toLong(), 1)
                                .also { it.attach(batchManager) }
                        }
                    val countBetter = batchManager.zeros(Shape(batchSize.toLong()), DataType.INT64)
                    var chunkStart = 0
                    while (chunkStart < numEntities.toInt()) {
                        val chunkEnd = minOf(chunkStart + chunkSize, numEntities.toInt())
                        entities.get(NDIndex("$chunkStart:$chunkEnd, :")).use { entityChunk ->
                            val eR = entityChunk.get(rIndex)
                            val eI = entityChunk.get(iIndex)
                            val eJ = entityChunk.get(jIndex)
                            val eK = entityChunk.get(kIndex)
                            eR.use { er ->
                                eI.use { ei ->
                                    eJ.use { ej ->
                                        eK.use { ek ->
                                            val scores =
                                                aR.matMul(er.transpose())
                                                    .add(aI.matMul(ei.transpose()))
                                                    .add(aJ.matMul(ej.transpose()))
                                                    .add(aK.matMul(ek.transpose()))
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
                        chunkStart = chunkEnd
                    }
                    val ranksNd = countBetter.add(1)
                    val batchRanks = ranksNd.toLongArray()
                    ranksNd.close()
                    for (rank in batchRanks) {
                        ranks.add(rank.toInt())
                    }
                }
            }
            start = end
        }
        return ranks
    }

    /**
     * Compute per-instance ranks for the ComplEx model over the class's inputList.
     *
     * This evaluates each example by scoring the true entity against all candidate entities
     * (processed in chunks) and counting how many candidates score better, producing a
     * 1-based rank for each example.
     *
     * @param evalBatchSize Number of examples processed together in an outer batch.
     * @param entityChunkSize Number of candidate entities processed per inner chunk.
     * @param mode Whether to predict TAIL or HEAD for each input (affects which columns are fixed).
     * @param buildBatch Callback that builds an EvalBatch for the half-open index range [start, end).
     * @throws IllegalArgumentException If the model's entity embedding dimension is not even.
     * @return A list of 1-based ranks (one entry per inputList item) where a value of `1` means the true entity ranked first.
     */
    private fun computeRanksComplEx(
        evalBatchSize: Int,
        entityChunkSize: Int,
        mode: EvalMode,
        buildBatch: (start: Int, end: Int) -> EvalBatch,
    ): List<Int> {
        val model = complEx ?: return emptyList()
        val ranks = mutableListOf<Int>()
        val entities = model.getEntities()
        val edges = model.getEdges()
        val embDim2 = entities.shape[1].toInt()
        require(embDim2 % 2 == 0) { "ComplEx embeddings must have even dimension (found $embDim2)." }
        val halfDim = embDim2 / 2
        val realIndex = NDIndex(":, 0:$halfDim")
        val imagIndex = NDIndex(":, $halfDim:")
        val chunkSize = maxOf(1, entityChunkSize)
        var start = 0
        while (start < inputList.size) {
            val end = minOf(start + evalBatchSize, inputList.size)
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
                    while (chunkStart < numEntities.toInt()) {
                        val chunkEnd = minOf(chunkStart + chunkSize, numEntities.toInt())
                        entities.get(NDIndex("$chunkStart:$chunkEnd, :")).use { entityChunk ->
                            val eRe = entityChunk.get(realIndex)
                            val eIm = entityChunk.get(imagIndex)
                            eRe.use { er ->
                                eIm.use { ei ->
                                    er.transpose().use { erT ->
                                        ei.transpose().use { eiT ->
                                            val scores = a.matMul(erT).add(b.matMul(eiT))
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
                        chunkStart = chunkEnd
                    }
                    val ranksNd = countBetter.add(1)
                    val batchRanks = ranksNd.toLongArray()
                    ranksNd.close()
                    for (rank in batchRanks) {
                        ranks.add(rank.toInt())
                    }
                }
            }
            start = end
        }
        return ranks
    }

    /**
     * Compute ranked positions of true entities using the TransR model for the current input list.
     *
     * For each input item this projects entities with the relation-specific matrix, scores all candidate
     * entities in chunks, counts how many candidates score strictly better (or worse when
     * `higherIsBetter` is true) than the true entity, and produces a one-based rank.
     *
     * @param evalBatchSize Number of input triples processed per outer batch.
     * @param entityChunkSize Number of candidate entities evaluated per inner chunk.
     * @param mode Whether to evaluate HEAD or TAIL prediction.
     * @param buildBatch Callback that builds an EvalBatch for the input slice [start, end).
     * @return A list of one-based ranks (count of strictly better-scoring candidates + 1) corresponding to each input triple.
     */
    private fun computeRanksTransR(
        evalBatchSize: Int,
        entityChunkSize: Int,
        mode: EvalMode,
        buildBatch: (start: Int, end: Int) -> EvalBatch,
    ): List<Int> {
        val model = transR ?: return emptyList()
        val ranks = mutableListOf<Int>()
        val entities = model.getEntities()
        val edges = model.getEdges()
        val matrix = model.getMatrix()
        val chunkSize = maxOf(1, entityChunkSize)
        var start = 0
        while (start < inputList.size) {
            val end = minOf(start + evalBatchSize, inputList.size)
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
                        manager.create(idxArr).use { idxNd ->
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
                                    manager.create(longArrayOf(relId)).use { relIdNd ->
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
                                                                            while (chunkStart < numEntities.toInt()) {
                                                                                val chunkEnd =
                                                                                    minOf(
                                                                                        chunkStart + chunkSize,
                                                                                        numEntities.toInt(),
                                                                                    )
                                                                                entities.get(NDIndex("$chunkStart:$chunkEnd, :")).use { entityChunk ->
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
                                                                                                    val scores = d.abs().sum(intArrayOf(2))
                                                                                                    scores.use { sc ->
                                                                                                        val countBetterNd =
                                                                                                            if (higherIsBetter) {
                                                                                                                sc.gt(ts2d).sum(intArrayOf(1))
                                                                                                            } else {
                                                                                                                sc.lt(ts2d).sum(intArrayOf(1))
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
                        ranks.add((count + 1L).toInt())
                    }
                }
            }
            start = end
        }
        return ranks
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
    private fun computeRanksChunked(
        evalBatchSize: Int,
        entityChunkSize: Int,
        buildBatch: (start: Int, end: Int) -> EvalBatch,
        buildTrueInput: (basePair: NDArray, trueEntityCol: NDArray) -> NDArray,
        buildChunkInput: (basePair: NDArray, entityChunk: NDArray) -> ChunkInput,
    ): List<Int> {
        val ranks = mutableListOf<Int>()
        var start = 0
        while (start < inputList.size) {
            val end = minOf(start + evalBatchSize, inputList.size)
            val batch = buildBatch(start, end)
            batch.use {
                val trueInput = buildTrueInput(batch.basePair, batch.trueEntityCol)
                val trueScores = predictor.predict(NDList(trueInput)).singletonOrThrow()
                val trueScores2d = trueScores.reshape(batch.batchSize.toLong(), 1)
                val countBetter = manager.zeros(Shape(batch.batchSize.toLong()), DataType.INT64)
                try {
                    val chunkSize = maxOf(1, entityChunkSize)
                    var chunkStart = 0
                    while (chunkStart < numEntities.toInt()) {
                        val chunkEnd = minOf(chunkStart + chunkSize, numEntities.toInt())
                        val entityChunk =
                            manager.arange(chunkStart, chunkEnd, 1, DataType.INT64)
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
                            ranks.add(rank.toInt())
                        }
                    } finally {
                        ranksNd.close()
                    }
                } finally {
                    countBetter.close()
                    trueScores2d.close()
                    trueScores.close()
                    trueInput.close()
                }
            }
            start = end
        }
        return ranks
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
    private fun computeMetrics(ranks: List<Int>): Map<String, Float> {
        val total = ranks.size.toFloat()
        if (total == 0f) {
            return mapOf(
                "HIT@1" to 0f,
                "HIT@10" to 0f,
                "HIT@100" to 0f,
                "MRR" to 0f,
            )
        }
        return mapOf(
            "HIT@1" to ranks.count { it <= 1 } / total,
            "HIT@10" to ranks.count { it <= 10 } / total,
            "HIT@100" to ranks.count { it <= 100 } / total,
            "MRR" to ranks.sumOf { 1.0 / it }.toFloat() / total,
        )
    }

    /**
     * Computes evaluation metrics for tail prediction.
     *
     * @return A map containing the evaluation metrics: HIT@1, HIT@10, HIT@100, and MRR.
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
            val trueTailCol = manager.create(tailIds).reshape(batchSize.toLong(), 1)
            EvalBatch(
                basePair = headRel,
                trueIds = tailIds,
                trueEntityCol = trueTailCol,
                batchSize = batchSize,
                closeables = listOf(trueTailCol, headRel, headCol, relCol),
            )
        }

        val ranks =
            when {
                transE != null ->
                    computeRanksTransE(evalBatchSize, entityChunkSize, EvalMode.TAIL, buildBatch)
                transR != null ->
                    computeRanksTransR(evalBatchSize, entityChunkSize, EvalMode.TAIL, buildBatch)
                distMult != null ->
                    computeRanksDistMult(evalBatchSize, entityChunkSize, EvalMode.TAIL, buildBatch)
                rotatE != null ->
                    computeRanksRotatE(evalBatchSize, entityChunkSize, EvalMode.TAIL, buildBatch)
                quatE != null ->
                    computeRanksQuatE(evalBatchSize, entityChunkSize, EvalMode.TAIL, buildBatch)
                complEx != null ->
                    computeRanksComplEx(evalBatchSize, entityChunkSize, EvalMode.TAIL, buildBatch)
                else ->
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
            }

        return computeMetrics(ranks)
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
            val trueHeadCol = manager.create(headIds).reshape(batchSize.toLong(), 1)
            EvalBatch(
                basePair = relTail,
                trueIds = headIds,
                trueEntityCol = trueHeadCol,
                batchSize = batchSize,
                closeables = listOf(trueHeadCol, relTail, relCol, tailCol),
            )
        }

        val ranks =
            when {
                transE != null ->
                    computeRanksTransE(evalBatchSize, entityChunkSize, EvalMode.HEAD, buildBatch)
                transR != null ->
                    computeRanksTransR(evalBatchSize, entityChunkSize, EvalMode.HEAD, buildBatch)
                distMult != null ->
                    computeRanksDistMult(evalBatchSize, entityChunkSize, EvalMode.HEAD, buildBatch)
                rotatE != null ->
                    computeRanksRotatE(evalBatchSize, entityChunkSize, EvalMode.HEAD, buildBatch)
                quatE != null ->
                    computeRanksQuatE(evalBatchSize, entityChunkSize, EvalMode.HEAD, buildBatch)
                complEx != null ->
                    computeRanksComplEx(evalBatchSize, entityChunkSize, EvalMode.HEAD, buildBatch)
                else ->
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

        return computeMetrics(ranks)
    }

    /**
     * Closes the NDManager instance.
     */
    fun close() {
        manager.close()
    }
}
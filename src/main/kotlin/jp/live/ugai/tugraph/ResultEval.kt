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
                val baseCol0 = basePair.get(col0Index)
                val baseCol1 = basePair.get(col1Index)
                val trueIdsFlat = trueEntityCol.reshape(batchSize.toLong())
                try {
                    val baseEmb =
                        when (mode) {
                            EvalMode.TAIL -> {
                                val heads = entities.get(baseCol0)
                                val rels = edges.get(baseCol1)
                                try {
                                    heads.add(rels)
                                } finally {
                                    heads.close()
                                    rels.close()
                                }
                            }
                            EvalMode.HEAD -> {
                                val rels = edges.get(baseCol0)
                                val tails = entities.get(baseCol1)
                                try {
                                    tails.sub(rels)
                                } finally {
                                    tails.close()
                                    rels.close()
                                }
                            }
                        }
                    val trueEmb = entities.get(trueIdsFlat)
                    var trueScore: NDArray? = null
                    var diffTrue: NDArray? = null
                    var absTrue: NDArray? = null
                    try {
                        diffTrue = baseEmb.sub(trueEmb)
                        absTrue = diffTrue.abs()
                        trueScore = absTrue.sum(intArrayOf(1)).reshape(batchSize.toLong(), 1)
                    } finally {
                        absTrue?.close()
                        diffTrue?.close()
                    }
                    val countBetter = baseEmb.manager.zeros(Shape(batchSize.toLong()), DataType.INT64)
                    try {
                        val baseEmbExp = baseEmb.expandDims(1)
                        var chunkStart = 0
                        try {
                            while (chunkStart < numEntities.toInt()) {
                                val chunkEnd = minOf(chunkStart + chunkSize, numEntities.toInt())
                                val entityChunk = entities.get(NDIndex("$chunkStart:$chunkEnd, :"))
                                try {
                                    val entityExp = entityChunk.expandDims(0)
                                    var diff: NDArray? = null
                                    var abs: NDArray? = null
                                    var scores: NDArray? = null
                                    try {
                                        diff = baseEmbExp.sub(entityExp)
                                        abs = diff.abs()
                                        scores = abs.sum(intArrayOf(2))
                                        val countBetterNd =
                                            if (higherIsBetter) {
                                                scores.gt(trueScore!!).sum(intArrayOf(1))
                                            } else {
                                                scores.lt(trueScore!!).sum(intArrayOf(1))
                                            }
                                        try {
                                            countBetter.addi(countBetterNd)
                                        } finally {
                                            countBetterNd.close()
                                        }
                                    } finally {
                                        scores?.close()
                                        abs?.close()
                                        diff?.close()
                                        entityExp.close()
                                    }
                                } finally {
                                    entityChunk.close()
                                }
                                chunkStart = chunkEnd
                            }
                        } finally {
                            baseEmbExp.close()
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
                        trueScore?.close()
                        trueEmb.close()
                        baseEmb.close()
                    }
                } finally {
                    trueIdsFlat.close()
                    baseCol1.close()
                    baseCol0.close()
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
                val baseCol0 = basePair.get(col0Index)
                val baseCol1 = basePair.get(col1Index)
                val trueIdsFlat = trueEntityCol.reshape(batchSize.toLong())
                try {
                    val relIds = if (mode == EvalMode.TAIL) baseCol1 else baseCol0
                    val fixedIds = if (mode == EvalMode.TAIL) baseCol0 else baseCol1
                    val rels = edges.get(relIds)
                    val fixed = entities.get(fixedIds)
                    val baseMul = fixed.mul(rels)
                    val trueEmb = entities.get(trueIdsFlat)
                    var trueScore: NDArray? = null
                    var trueMul: NDArray? = null
                    var trueSum: NDArray? = null
                    try {
                        trueMul = baseMul.mul(trueEmb)
                        trueSum = trueMul.sum(intArrayOf(1))
                        trueScore = trueSum.reshape(batchSize.toLong(), 1)
                    } finally {
                        trueSum?.close()
                        trueMul?.close()
                    }
                    val countBetter = baseMul.manager.zeros(Shape(batchSize.toLong()), DataType.INT64)
                    try {
                        var chunkStart = 0
                        while (chunkStart < numEntities.toInt()) {
                            val chunkEnd = minOf(chunkStart + chunkSize, numEntities.toInt())
                            val entityChunk = entities.get(NDIndex("$chunkStart:$chunkEnd, :"))
                            val entityChunkT = entityChunk.transpose()
                            try {
                                val scores = baseMul.matMul(entityChunkT)
                                val countBetterNd =
                                    if (higherIsBetter) {
                                        scores.gt(trueScore).sum(intArrayOf(1))
                                    } else {
                                        scores.lt(trueScore).sum(intArrayOf(1))
                                    }
                                try {
                                    countBetter.addi(countBetterNd)
                                } finally {
                                    countBetterNd.close()
                                }
                                scores.close()
                            } finally {
                                entityChunkT.close()
                                entityChunk.close()
                            }
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
                        trueScore?.close()
                        trueEmb.close()
                        baseMul.close()
                        fixed.close()
                        rels.close()
                    }
                } finally {
                    trueIdsFlat.close()
                    baseCol1.close()
                    baseCol0.close()
                }
            }
            start = end
        }
        return ranks
    }

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
                val baseCol0 = basePair.get(col0Index)
                val baseCol1 = basePair.get(col1Index)
                val trueIdsFlat = trueEntityCol.reshape(batchSize.toLong())
                try {
                    val relIds = if (mode == EvalMode.TAIL) baseCol1 else baseCol0
                    val fixedIds = if (mode == EvalMode.TAIL) baseCol0 else baseCol1
                    val rels = edges.get(relIds)
                    val fixed = entities.get(fixedIds)
                    val rRe = rels.get(realIndex)
                    val rIm = rels.get(imagIndex)
                    val fRe = fixed.get(realIndex)
                    val fIm = fixed.get(imagIndex)
                    val a: NDArray
                    val b: NDArray
                    var term1: NDArray? = null
                    var term2: NDArray? = null
                    var term3: NDArray? = null
                    var term4: NDArray? = null
                    try {
                        if (mode == EvalMode.TAIL) {
                            term1 = fRe.mul(rRe)
                            term2 = fIm.mul(rIm)
                            a = term1.sub(term2)
                            term3 = fRe.mul(rIm)
                            term4 = fIm.mul(rRe)
                            b = term3.add(term4)
                        } else {
                            term1 = rRe.mul(fRe)
                            term2 = rIm.mul(fIm)
                            a = term1.add(term2)
                            term3 = rRe.mul(fIm)
                            term4 = rIm.mul(fRe)
                            b = term3.sub(term4)
                        }
                    } finally {
                        term4?.close()
                        term3?.close()
                        term2?.close()
                        term1?.close()
                    }
                    val trueEmb = entities.get(trueIdsFlat)
                    val tRe = trueEmb.get(realIndex)
                    val tIm = trueEmb.get(imagIndex)
                    var trueScore: NDArray? = null
                    var trueMulRe: NDArray? = null
                    var trueMulIm: NDArray? = null
                    var trueSum: NDArray? = null
                    try {
                        trueMulRe = a.mul(tRe)
                        trueMulIm = b.mul(tIm)
                        trueSum = trueMulRe.add(trueMulIm)
                        trueScore = trueSum.sum(intArrayOf(1)).reshape(batchSize.toLong(), 1)
                    } finally {
                        trueSum?.close()
                        trueMulIm?.close()
                        trueMulRe?.close()
                    }
                    val countBetter = a.manager.zeros(Shape(batchSize.toLong()), DataType.INT64)
                    try {
                        var chunkStart = 0
                        while (chunkStart < numEntities.toInt()) {
                            val chunkEnd = minOf(chunkStart + chunkSize, numEntities.toInt())
                            val entityChunk = entities.get(NDIndex("$chunkStart:$chunkEnd, :"))
                            val eRe = entityChunk.get(realIndex)
                            val eIm = entityChunk.get(imagIndex)
                            val eReT = eRe.transpose()
                            val eImT = eIm.transpose()
                            try {
                                val scoresRe = a.matMul(eReT)
                                val scoresIm = b.matMul(eImT)
                                val scores = scoresRe.add(scoresIm)
                                val countBetterNd =
                                    if (higherIsBetter) {
                                        scores.gt(trueScore).sum(intArrayOf(1))
                                    } else {
                                        scores.lt(trueScore).sum(intArrayOf(1))
                                    }
                                try {
                                    countBetter.addi(countBetterNd)
                                } finally {
                                    countBetterNd.close()
                                }
                                scores.close()
                                scoresIm.close()
                                scoresRe.close()
                            } finally {
                                eImT.close()
                                eReT.close()
                                eIm.close()
                                eRe.close()
                                entityChunk.close()
                            }
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
                        trueScore?.close()
                        tIm.close()
                        tRe.close()
                        trueEmb.close()
                        b.close()
                        a.close()
                        fIm.close()
                        fRe.close()
                        rIm.close()
                        rRe.close()
                        fixed.close()
                        rels.close()
                    }
                } finally {
                    trueIdsFlat.close()
                    baseCol1.close()
                    baseCol0.close()
                }
            }
            start = end
        }
        return ranks
    }

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
                val baseCol0 = basePair.get(col0Index)
                val baseCol1 = basePair.get(col1Index)
                val trueIdsFlat = trueEntityCol.reshape(batchSize.toLong())
                try {
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
                        val idxNd = manager.create(idxArr)
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
                        val relIdNd = manager.create(longArrayOf(relId))
                        try {
                            val heads = entities.get(headIdsNd)
                            val tails = entities.get(tailIdsNd)
                            val relEmb = edges.get(relIdNd)
                            val matrixRel = matrix.get(relIdNd).reshape(relEmb.shape[1], heads.shape[1])
                            val matrixT = matrixRel.transpose()
                            val headProj = heads.matMul(matrixT)
                            val tailProj = tails.matMul(matrixT)
                            val base =
                                if (mode == EvalMode.TAIL) {
                                    headProj.add(relEmb)
                                } else {
                                    relEmb.sub(tailProj)
                                }
                            var trueScore: NDArray? = null
                            var trueDiff: NDArray? = null
                            var trueAbs: NDArray? = null
                            try {
                                trueDiff =
                                    if (mode == EvalMode.TAIL) {
                                        base.sub(tailProj)
                                    } else {
                                        base.add(headProj)
                                    }
                                trueAbs = trueDiff.abs()
                                trueScore = trueAbs.sum(intArrayOf(1))
                            } finally {
                                trueAbs?.close()
                                trueDiff?.close()
                            }
                            val trueScore2d = trueScore!!.reshape(idxArr.size.toLong(), 1)
                            val countBetter = base.manager.zeros(Shape(idxArr.size.toLong()), DataType.INT64)
                            try {
                                var chunkStart = 0
                                while (chunkStart < numEntities.toInt()) {
                                    val chunkEnd = minOf(chunkStart + chunkSize, numEntities.toInt())
                                    val entityChunk = entities.get(NDIndex("$chunkStart:$chunkEnd, :"))
                                    val projChunk = entityChunk.matMul(matrixT)
                                    val projExp = projChunk.expandDims(0)
                                    val baseExp = base.expandDims(1)
                                    var scores: NDArray? = null
                                    var abs: NDArray? = null
                                    var diff: NDArray? = null
                                    try {
                                        diff =
                                            if (mode == EvalMode.TAIL) {
                                                baseExp.sub(projExp)
                                            } else {
                                                baseExp.add(projExp)
                                            }
                                        abs = diff.abs()
                                        scores = abs.sum(intArrayOf(2))
                                        val countBetterNd =
                                            if (higherIsBetter) {
                                                scores.gt(trueScore2d).sum(intArrayOf(1))
                                            } else {
                                                scores.lt(trueScore2d).sum(intArrayOf(1))
                                            }
                                        try {
                                            countBetter.addi(countBetterNd)
                                        } finally {
                                            countBetterNd.close()
                                        }
                                    } finally {
                                        scores?.close()
                                        abs?.close()
                                        diff?.close()
                                        baseExp.close()
                                        projExp.close()
                                        projChunk.close()
                                        entityChunk.close()
                                    }
                                    chunkStart = chunkEnd
                                }
                                val subsetCounts = countBetter.toLongArray()
                                for (i in idxArr.indices) {
                                    countBetterAll[idxArr[i]] = subsetCounts[i]
                                }
                            } finally {
                                countBetter.close()
                                trueScore2d.close()
                                trueScore.close()
                                base.close()
                                tailProj.close()
                                headProj.close()
                                matrixT.close()
                                matrixRel.close()
                                relEmb.close()
                                tails.close()
                                heads.close()
                            }
                        } finally {
                            relIdNd.close()
                            tailIdsNd.close()
                            headIdsNd.close()
                            idxNd.close()
                        }
                    }
                    for (count in countBetterAll) {
                        ranks.add((count + 1L).toInt())
                    }
                } finally {
                    trueIdsFlat.close()
                    baseCol1.close()
                    baseCol0.close()
                }
            }
            start = end
        }
        return ranks
    }

    /**
     * Computes rank positions of the true entity for each input triple across batches.
     *
     * The rank for an item is 1 plus the number of candidate entities whose score is strictly
     * better than the true entity's score. "Strictly better" means greater than the true score
     * when `higherIsBetter` is true, or less than the true score when `higherIsBetter` is false.
     *
     * @param evalBatchSize Maximum number of input triples processed per batch.
     * @param buildBatch Function that builds an EvalBatch for the half-open range [start, end).
     * @return A list of ranks (one per input triple in inputList order).
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
     * Computes evaluation metrics for tail prediction.
     *
     * @return A map containing the evaluation metrics: HIT@1, HIT@10, HIT@100, and MRR.
     */
    fun getTailResult(): Map<String, Float> {
        val result = mutableMapOf<String, Float>()
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
                headArr[i] = triple[0]
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

        // Compute metrics
        val total = ranks.size.toFloat()
        result["HIT@1"] = ranks.count { it <= 1 } / total
        result["HIT@10"] = ranks.count { it <= 10 } / total
        result["HIT@100"] = ranks.count { it <= 100 } / total
        result["MRR"] = ranks.sumOf { 1.0 / it }.toFloat() / total

        return result
    }

    /**
     * Computes evaluation metrics for head prediction.
     *
     * @return A map containing the evaluation metrics: HIT@1, HIT@10, HIT@100, and MRR.
     */
    fun getHeadResult(): Map<String, Float> {
        val result = mutableMapOf<String, Float>()
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
                tailArr[i] = triple[2]
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

        // Compute metrics
        val total = ranks.size.toFloat()
        result["HIT@1"] = ranks.count { it <= 1 } / total
        result["HIT@10"] = ranks.count { it <= 10 } / total
        result["HIT@100"] = ranks.count { it <= 100 } / total
        result["MRR"] = ranks.sumOf { 1.0 / it }.toFloat() / total

        return result
    }

    /**
     * Closes the NDManager instance.
     */
    fun close() {
        manager.close()
    }
}

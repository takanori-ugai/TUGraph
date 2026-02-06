package jp.live.ugai.tugraph.eval

import ai.djl.inference.Predictor
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import jp.live.ugai.tugraph.TransE

class ResultEvalTransE(
    inputList: List<LongArray>,
    manager: NDManager,
    predictor: Predictor<NDList, NDList>,
    numEntities: Long,
    val transE: TransE,
    higherIsBetter: Boolean = false,
    filtered: Boolean = false,
) : ResultEval(inputList, manager, predictor, numEntities, higherIsBetter, filtered) {
    private val col0Index = NDIndex(":, 0")
    private val col1Index = NDIndex(":, 1")

    /**
     * Compute 1-based ranks of the true entities for each input pair according to TransE scores.
     *
     * Processes inputs in evaluation batches and iterates candidate entities in chunks to limit memory usage.
     * For each pair it builds the base embedding according to `mode`, computes the L1 distance to the true entity,
     * compares that true score against all candidate entity scores (respecting `higherIsBetter`), counts how many
     * candidates are ranked better, and produces a rank by adding 1 to that count. Returns an empty array if there
     * are no inputs.
     *
     * @param evalBatchSize Number of evaluation examples processed per batch.
     * @param entityChunkSize Maximum number of entities loaded at once when comparing candidates.
     * @param mode Whether to score by replacing the head or tail (affects how the base embedding is formed).
     * @param buildBatch Function that builds an EvalBatch for the input slice [start, end).
     * @return An IntArray of 1-based ranks corresponding to `inputList` order; length equals the number of processed inputs. 
     */
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
        val entities = transE.getEntities()
        val edges = transE.getEdges()
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
                        baseEmb
                            .sub(trueEmb)
                            .abs()
                            .sum(intArrayOf(1))
                            .reshape(batchSize.toLong(), 1)
                            .also { it.attach(batchManager) }
                    val countBetter = batchManager.zeros(Shape(batchSize.toLong()), DataType.INT64)
                    val baseEmbExp = baseEmb.expandDims(1).also { it.attach(batchManager) }
                    var chunkStart = 0
                    while (chunkStart < numEntitiesInt) {
                        val chunkEnd = minOf(chunkStart + chunkSize, numEntitiesInt)
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
                        ranks[outIndex++] = rank.toInt()
                    }
                }
            }
            start = end
        }
        return if (outIndex == totalSize) ranks else ranks.copyOf(outIndex)
    }
}
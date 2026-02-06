package jp.live.ugai.tugraph

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex

internal object EvalQuatE {
    data class Parts(
        val r: NDArray,
        val i: NDArray,
        val j: NDArray,
        val k: NDArray,
    )

    fun sliceQuat(
        emb: NDArray,
        rIndex: NDIndex,
        iIndex: NDIndex,
        jIndex: NDIndex,
        kIndex: NDIndex,
        batchManager: NDManager,
    ): Parts {
        val r = emb.get(rIndex).also { it.attach(batchManager) }
        val i = emb.get(iIndex).also { it.attach(batchManager) }
        val j = emb.get(jIndex).also { it.attach(batchManager) }
        val k = emb.get(kIndex).also { it.attach(batchManager) }
        return Parts(r, i, j, k)
    }

    /**
     * Compute the quaternion product of a fixed quaternion and a relation quaternion for tail composition.
     *
     * @param fixed Quaternion components (r, i, j, k) representing the fixed quaternion.
     * @param rel Quaternion components (r, i, j, k) representing the relation quaternion.
     * @param batchManager NDManager used to attach the resulting NDArrays.
     * @return Parts containing the resulting quaternion components `(r, i, j, k)` of the product, each attached to `batchManager`.
     */
    fun computeATail(
        fixed: Parts,
        rel: Parts,
        batchManager: NDManager,
    ): Parts {
        val hrR =
            fixed.r
                .mul(rel.r)
                .sub(fixed.i.mul(rel.i))
                .sub(fixed.j.mul(rel.j))
                .sub(fixed.k.mul(rel.k))
        val hrI =
            fixed.r
                .mul(rel.i)
                .add(fixed.i.mul(rel.r))
                .add(fixed.j.mul(rel.k))
                .sub(fixed.k.mul(rel.j))
        val hrJ =
            fixed.r
                .mul(rel.j)
                .sub(fixed.i.mul(rel.k))
                .add(fixed.j.mul(rel.r))
                .add(fixed.k.mul(rel.i))
        val hrK =
            fixed.r
                .mul(rel.k)
                .add(fixed.i.mul(rel.j))
                .sub(fixed.j.mul(rel.i))
                .add(fixed.k.mul(rel.r))
        val aR = hrR.also { it.attach(batchManager) }
        val aI = hrI.also { it.attach(batchManager) }
        val aJ = hrJ.also { it.attach(batchManager) }
        val aK = hrK.also { it.attach(batchManager) }
        return Parts(aR, aI, aJ, aK)
    }

    fun computeAHead(
        rel: Parts,
        trueIdsFlat: NDArray,
        entities: NDArray,
        rIndex: NDIndex,
        iIndex: NDIndex,
        jIndex: NDIndex,
        kIndex: NDIndex,
        batchManager: NDManager,
    ): Parts {
        val tEmb = entities.get(trueIdsFlat).also { it.attach(batchManager) }
        val t = sliceQuat(tEmb, rIndex, iIndex, jIndex, kIndex, batchManager)
        val aR =
            rel.r.mul(t.r).add(rel.i.mul(t.i)).add(rel.j.mul(t.j)).add(rel.k.mul(t.k)).also {
                it.attach(batchManager)
            }
        val aI =
            rel.r.mul(t.i).sub(rel.i.mul(t.r)).add(rel.j.mul(t.k)).sub(rel.k.mul(t.j)).also {
                it.attach(batchManager)
            }
        val aJ =
            rel.r.mul(t.j).sub(rel.j.mul(t.r)).add(rel.k.mul(t.i)).sub(rel.i.mul(t.k)).also {
                it.attach(batchManager)
            }
        val aK =
            rel.r.mul(t.k).sub(rel.k.mul(t.r)).add(rel.i.mul(t.j)).sub(rel.j.mul(t.i)).also {
                it.attach(batchManager)
            }
        return Parts(aR, aI, aJ, aK)
    }

    /**
     * Compute the true-tail score for a batch using accumulated head quaternion components.
     *
     * @param a Accumulated quaternion components (r, i, j, k) for the head.
     * @param trueIdsFlat 1-D indices selecting the true tail embeddings from `entities`.
     * @param entities NDArray containing entity embeddings organized with quaternion components.
     * @param rIndex NDIndex selecting the real component within an embedding.
     * @param iIndex NDIndex selecting the first imaginary component within an embedding.
     * @param jIndex NDIndex selecting the second imaginary component within an embedding.
     * @param kIndex NDIndex selecting the third imaginary component within an embedding.
     * @param batchSize Number of rows in the resulting score (output shape will be (batchSize, 1)).
     * @return NDArray of shape (batchSize, 1) containing the score for each true tail, computed as the sum of elementwise products across quaternion components (r, i, j, k).
     */
    fun computeTrueScoreTail(
        a: Parts,
        trueIdsFlat: NDArray,
        entities: NDArray,
        rIndex: NDIndex,
        iIndex: NDIndex,
        jIndex: NDIndex,
        kIndex: NDIndex,
        batchSize: Int,
        batchManager: NDManager,
    ): NDArray {
        val trueEmb = entities.get(trueIdsFlat).also { it.attach(batchManager) }
        val t = sliceQuat(trueEmb, rIndex, iIndex, jIndex, kIndex, batchManager)
        return a.r
            .mul(t.r)
            .add(a.i.mul(t.i))
            .add(a.j.mul(t.j))
            .add(a.k.mul(t.k))
            .sum(intArrayOf(1))
            .reshape(batchSize.toLong(), 1)
            .also { it.attach(batchManager) }
    }

    /**
     * Compute scores between an accumulated quaternion head `a` and true head entity embeddings selected by `trueIdsFlat`.
     *
     * The score for each pair is the sum over component-wise products: a.r*h.r + a.i*h.i + a.j*h.j + a.k*h.k, returned with shape (batchSize, 1).
     *
     * @param a Accumulated quaternion head components (r, i, j, k).
     * @param trueIdsFlat NDArray of indices selecting true head embeddings from `entities`.
     * @param entities NDArray containing entity embeddings arranged so quaternion components can be sliced by the provided indices.
     * @param rIndex NDIndex selecting the real component slice within entity embeddings.
     * @param iIndex NDIndex selecting the i component slice within entity embeddings.
     * @param jIndex NDIndex selecting the j component slice within entity embeddings.
     * @param kIndex NDIndex selecting the k component slice within entity embeddings.
     * @param batchSize Number of rows in the resulting score (batch size).
     * @param batchManager NDManager to attach created NDArrays.
     * @return NDArray of shape (batchSize, 1) with the computed head scores.
     */
    fun computeTrueScoreHead(
        a: Parts,
        trueIdsFlat: NDArray,
        entities: NDArray,
        rIndex: NDIndex,
        iIndex: NDIndex,
        jIndex: NDIndex,
        kIndex: NDIndex,
        batchSize: Int,
        batchManager: NDManager,
    ): NDArray {
        val trueEmb = entities.get(trueIdsFlat).also { it.attach(batchManager) }
        val h = sliceQuat(trueEmb, rIndex, iIndex, jIndex, kIndex, batchManager)
        return a.r
            .mul(h.r)
            .add(a.i.mul(h.i))
            .add(a.j.mul(h.j))
            .add(a.k.mul(h.k))
            .sum(intArrayOf(1))
            .reshape(batchSize.toLong(), 1)
            .also { it.attach(batchManager) }
    }

    /**
     * Accumulates counts of entities whose scores are better than the provided true scores, processing entities in chunks.
     *
     * Processes the `entities` array in chunks of size `chunkSize`. For each chunk this function:
     * - extracts quaternion components using `rIndex`, `iIndex`, `jIndex`, `kIndex`,
     * - computes scores for each entity in the chunk against the quaternion `a`,
     * - compares those scores to `trueScore` using `higherIsBetter` to decide the comparison direction,
     * - increments `countBetter` in place with the number of better scores per batch row.
     *
     * @param entities Entity embeddings arranged with quaternion components (shape: [numEntities, ...]).
     * @param rIndex NDIndex selector for the real component within an entity embedding.
     * @param iIndex NDIndex selector for the first imaginary component within an entity embedding.
     * @param jIndex NDIndex selector for the second imaginary component within an entity embedding.
     * @param kIndex NDIndex selector for the third imaginary component within an entity embedding.
     * @param a Quaternion parts (r, i, j, k) used to compute scores against entities.
     * @param trueScore NDArray of reference scores to compare against (one score per row of the current batch).
     * @param countBetter NDArray accumulator that is incremented in place with counts of better scores.
     * @param higherIsBetter If `true`, a higher score is considered better (use `>`); otherwise lower is better (use `<`).
     * @param chunkSize Number of entities to process per iteration to limit memory usage.
     * @param numEntitiesInt Total number of entities to evaluate.
     */
    fun accumulateCountBetter(
        entities: NDArray,
        rIndex: NDIndex,
        iIndex: NDIndex,
        jIndex: NDIndex,
        kIndex: NDIndex,
        a: Parts,
        trueScore: NDArray,
        countBetter: NDArray,
        higherIsBetter: Boolean,
        chunkSize: Int,
        numEntitiesInt: Int,
    ) {
        var chunkStart = 0
        while (chunkStart < numEntitiesInt) {
            val chunkEnd = minOf(chunkStart + chunkSize, numEntitiesInt)
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
                                    a.r
                                        .matMul(er.transpose())
                                        .add(a.i.matMul(ei.transpose()))
                                        .add(a.j.matMul(ej.transpose()))
                                        .add(a.k.matMul(ek.transpose()))
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
    }
}

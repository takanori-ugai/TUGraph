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

    fun computeATail(
        fixed: Parts,
        rel: Parts,
        batchManager: NDManager,
    ): Parts {
        val hrR = fixed.r.mul(rel.r).sub(fixed.i.mul(rel.i)).sub(fixed.j.mul(rel.j)).sub(fixed.k.mul(rel.k))
        val hrI = fixed.r.mul(rel.i).add(fixed.i.mul(rel.r)).add(fixed.j.mul(rel.k)).sub(fixed.k.mul(rel.j))
        val hrJ = fixed.r.mul(rel.j).sub(fixed.i.mul(rel.k)).add(fixed.j.mul(rel.r)).add(fixed.k.mul(rel.i))
        val hrK = fixed.r.mul(rel.k).add(fixed.i.mul(rel.j)).sub(fixed.j.mul(rel.i)).add(fixed.k.mul(rel.r))
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
        val aR = rel.r.mul(t.r).add(rel.i.mul(t.i)).add(rel.j.mul(t.j)).add(rel.k.mul(t.k)).also {
            it.attach(batchManager)
        }
        val aI = rel.r.mul(t.i).sub(rel.i.mul(t.r)).add(rel.j.mul(t.k)).sub(rel.k.mul(t.j)).also {
            it.attach(batchManager)
        }
        val aJ = rel.r.mul(t.j).sub(rel.j.mul(t.r)).add(rel.k.mul(t.i)).sub(rel.i.mul(t.k)).also {
            it.attach(batchManager)
        }
        val aK = rel.r.mul(t.k).sub(rel.k.mul(t.r)).add(rel.i.mul(t.j)).sub(rel.j.mul(t.i)).also {
            it.attach(batchManager)
        }
        return Parts(aR, aI, aJ, aK)
    }

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
        return a.r.mul(t.r).add(a.i.mul(t.i)).add(a.j.mul(t.j)).add(a.k.mul(t.k)).sum(intArrayOf(1))
            .reshape(batchSize.toLong(), 1)
            .also { it.attach(batchManager) }
    }

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
        return a.r.mul(h.r).add(a.i.mul(h.i)).add(a.j.mul(h.j)).add(a.k.mul(h.k)).sum(intArrayOf(1))
            .reshape(batchSize.toLong(), 1)
            .also { it.attach(batchManager) }
    }

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
                                    a.r.matMul(er.transpose())
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

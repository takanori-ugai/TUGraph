package jp.live.ugai.tugraph.train

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import jp.live.ugai.tugraph.NEGATIVE_RESAMPLE_CAP
import jp.live.ugai.tugraph.TRIPLE
import org.slf4j.Logger
import java.util.concurrent.ThreadLocalRandom

/**
 * Generates negative samples for a batch of triples.
 *
 * Uses optional Bernoulli sampling per relation and avoids existing triples via [inputSet].
 *
 * @param numEntities Total number of entities available for sampling.
 * @param bernoulliProb Relation-specific probabilities for head corruption.
 * @param inputSet Set of known triples used to avoid false negatives.
 * @param logger Logger for warnings when resampling fails.
 */
internal class NegativeSampler(
    private val numEntities: Long,
    private val bernoulliProb: Map<Long, Float>,
    private val inputSet: Set<TripleKey>,
    private val logger: Logger,
) {
    /**
     * Samples negative triples for the provided batch.
     *
     * @param manager NDManager used to allocate the output array.
     * @param input NDArray of triples to corrupt (shape [batchSize, 3] or equivalent flat size).
     * @param numNegatives Number of negatives per positive triple.
     * @param useBernoulli Whether to use relation-specific Bernoulli corruption.
     * @return NDArray of negatives with shape [batchSize * numNegatives, 3].
     */
    fun sample(
        manager: NDManager,
        input: NDArray,
        numNegatives: Int,
        useBernoulli: Boolean,
    ): NDArray {
        require(numEntities > 1L) { "numEntities must be > 1 for negative sampling, was $numEntities." }
        val numTriples = input.size() / TRIPLE
        require(numTriples <= Int.MAX_VALUE.toLong()) {
            "Batch exceeds Int indexing capacity (numTriples=$numTriples)."
        }
        val batchCount = numTriples.toInt()
        val totalNegatives = batchCount.toLong() * numNegatives.toLong()
        val totalElements = totalNegatives * TRIPLE
        require(totalElements <= Int.MAX_VALUE.toLong()) {
            "Negative sample array exceeds Int capacity ($totalElements elements)."
        }
        val negatives = LongArray(totalElements.toInt())
        val maxResample = NEGATIVE_RESAMPLE_CAP
        val flat = input.toLongArray()
        for (i in 0 until batchCount) {
            val base = i * TRIPLE.toInt()
            val fst = flat[base]
            val sec = flat[base + 1]
            val trd = flat[base + 2]
            val headProb = if (useBernoulli) bernoulliProb[sec] ?: 0.5f else 0.5f
            for (n in 0 until numNegatives) {
                var replaceHeadFlag = ThreadLocalRandom.current().nextFloat() < headProb
                var ran: Long
                var checkTriplet = TripleKey(0L, 0L, 0L)
                var attempts = 0
                var found = false
                while (attempts < maxResample) {
                    ran = ThreadLocalRandom.current().nextLong(numEntities)
                    if (replaceHeadFlag) {
                        if (ran == fst) {
                            attempts += 1
                            continue
                        }
                        checkTriplet = TripleKey(ran, sec, trd)
                    } else {
                        if (ran == trd) {
                            attempts += 1
                            continue
                        }
                        checkTriplet = TripleKey(fst, sec, ran)
                    }
                    if (!inputSet.contains(checkTriplet)) {
                        found = true
                        break
                    }
                    replaceHeadFlag = !replaceHeadFlag
                    attempts += 1
                }
                val negIndex = i * numNegatives + n
                val negBase = negIndex * TRIPLE.toInt()
                if (found) {
                    negatives[negBase] = checkTriplet.head
                    negatives[negBase + 1] = checkTriplet.rel
                    negatives[negBase + 2] = checkTriplet.tail
                } else {
                    negatives[negBase] = fst
                    negatives[negBase + 1] = sec
                    negatives[negBase + 2] = trd
                    logger.warn(
                        "Negative resample cap hit ({} attempts) for batch row {}.",
                        maxResample,
                        i,
                    )
                }
            }
        }
        return manager.create(negatives, Shape(totalNegatives, TRIPLE))
    }
}

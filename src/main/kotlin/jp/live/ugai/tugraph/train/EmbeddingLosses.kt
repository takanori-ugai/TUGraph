package jp.live.ugai.tugraph.train

import ai.djl.ndarray.NDArray
import ai.djl.nn.Activation
import ai.djl.training.ParameterStore
import ai.djl.training.Trainer
import jp.live.ugai.tugraph.HYPERCOMPLEX_BETA
import jp.live.ugai.tugraph.HYPERCOMPLEX_GAMMA
import jp.live.ugai.tugraph.HYPERCOMPLEX_LAMBDA1
import jp.live.ugai.tugraph.HYPERCOMPLEX_LAMBDA2
import jp.live.ugai.tugraph.HyperComplEx
import jp.live.ugai.tugraph.MATRYOSHKA_QUATE_DIMS
import jp.live.ugai.tugraph.MATRYOSHKA_QUATE_WEIGHTS
import jp.live.ugai.tugraph.MATRYOSHKA_ROTATE_DIMS
import jp.live.ugai.tugraph.MATRYOSHKA_ROTATE_WEIGHTS
import jp.live.ugai.tugraph.QuatE
import jp.live.ugai.tugraph.RotatE
import jp.live.ugai.tugraph.SELF_ADVERSARIAL_TEMP

internal class EmbeddingLosses(
    private val trainer: Trainer,
) {
    private val margin = 1.0f

    /**
     * Compute per-sample loss from positive and negative scores using either a similarity-based
     * objective or a hinge (margin) objective.
     *
     * When `useSimilarityLoss` is true, loss per sample is softplus(-pos) plus an aggregate of
     * softplus(neg) across negatives; the aggregate is the mean unless `useSelfAdversarial` is true,
     * in which case negatives are weighted by a softmax over `neg` scaled by `SELF_ADVERSARIAL_TEMP`.
     * When `useSimilarityLoss` is false, the hinge loss with margin `margin` is applied and averaged
     * across negatives; `higherIsBetter` controls the hinge direction.
     *
     * @param pos Positive scores shaped (batchSize,).
     * @param neg Negative scores shaped so they can be viewed as (batchSize, numNegatives).
     * @param useSimilarityLoss If true, use the softplus-based similarity loss; otherwise use hinge loss.
     * @param useSelfAdversarial If true and `useSimilarityLoss` is true, apply self-adversarial weighting to negatives.
     * @param higherIsBetter If true, higher scores indicate better predictions and affect hinge sign.
     * @return 1D NDArray of per-sample losses with length equal to `batchSize`.
     */
    fun computeLossFromScores(
        pos: NDArray,
        neg: NDArray,
        useSimilarityLoss: Boolean,
        useSelfAdversarial: Boolean,
        higherIsBetter: Boolean,
    ): NDArray =
        if (useSimilarityLoss) {
            // softplus(-pos) + softplus(neg)
            val posLoss =
                pos.neg().use { negPos ->
                    negPos.exp().use { expPos ->
                        expPos.add(1f).use { addPos ->
                            addPos.log()
                        }
                    }
                }
            val negLoss =
                neg.exp().use { expNeg ->
                    expNeg.add(1f).use { addNeg ->
                        addNeg.log()
                    }
                }
            val negAgg =
                if (useSelfAdversarial) {
                    neg.mul(SELF_ADVERSARIAL_TEMP).use { scaled ->
                        scaled.softmax(1).use { weights ->
                            negLoss.use { nLoss ->
                                weights.mul(nLoss).use { weighted ->
                                    weighted.sum(intArrayOf(1))
                                }
                            }
                        }
                    }
                } else {
                    negLoss.use { nLoss ->
                        nLoss.mean(intArrayOf(1))
                    }
                }
            posLoss.use { pLoss ->
                negAgg.use { nAgg ->
                    pLoss.add(nAgg)
                }
            }
        } else {
            val hinge =
                if (higherIsBetter) {
                    pos.expandDims(1).use { posExp ->
                        neg.sub(posExp).use { diff ->
                            diff.add(margin).use { shifted ->
                                shifted.maximum(0f)
                            }
                        }
                    }
                } else {
                    pos.expandDims(1).use { posExp ->
                        posExp.sub(neg).use { diff ->
                            diff.add(margin).use { shifted ->
                                shifted.maximum(0f)
                            }
                        }
                    }
                }
            hinge.use { it.mean(intArrayOf(1)) }
        }

    /**
     * Computes HyperComplEx loss as the sum of self-adversarial ranking loss, multi-space consistency loss,
     * and L2 regularization.
     */
    data class HyperLossResult(
        val loss: NDArray,
        val posScores: NDArray,
        val negScores: NDArray,
    )

    fun computeHyperComplExLoss(
        block: HyperComplEx,
        sample: NDArray,
        negativeSample: NDArray,
        numNegatives: Int,
        training: Boolean,
    ): NDArray {
        val result =
            computeHyperComplExLossWithScores(
                block,
                sample,
                negativeSample,
                numNegatives,
                training,
            )
        result.posScores.close()
        result.negScores.close()
        return result.loss
    }

    /**
     * Compute the HyperComplEx loss components for a batch and return the aggregated loss along with positive and negative scores.
     *
     * @param block The HyperComplEx scoring block used to compute scores and regularization.
     * @param sample NDArray of positive samples (shape: [batchSize, 3]).
     * @param negativeSample NDArray of negative samples (shape: [batchSize * numNegatives, 3]).
     * @param numNegatives Number of negative samples per positive sample.
     * @param training Whether to compute training-mode quantities (affects parameter/regularization behavior).
     * @return HyperLossResult containing:
     *   - `loss`: aggregated loss NDArray combining ranking, consistency, and regularization terms,
     *   - `posScores`: positive sample scores per input sample,
     *   - `negScores`: negative sample scores reshaped to [batchSize, numNegatives].
     */
    fun computeHyperComplExLossWithScores(
        block: HyperComplEx,
        sample: NDArray,
        negativeSample: NDArray,
        numNegatives: Int,
        training: Boolean,
    ): HyperLossResult {
        val device = sample.device
        val parameterStore = ParameterStore(trainer.manager, true)
        val posParts = block.scoreParts(sample, parameterStore, training)
        val negParts = block.scoreParts(negativeSample, parameterStore, training)
        try {
            val posScores = posParts.total
            val negScores = negParts.total.reshape(posScores.shape[0], numNegatives.toLong())

            val gamma = HYPERCOMPLEX_GAMMA
            val beta = HYPERCOMPLEX_BETA
            val posLogSig =
                posScores
                    .neg()
                    .add(gamma)
                    .use { shifted ->
                        Activation.sigmoid(shifted).use { sig ->
                            sig.log()
                        }
                    }
            val negLogSig =
                negScores
                    .sub(gamma)
                    .use { shifted ->
                        Activation.sigmoid(shifted).use { sig ->
                            sig.log()
                        }
                    }
            val posLoss = posLogSig.use { it.neg() }
            val weights = negScores.mul(beta).use { it.softmax(1) }
            val negLoss =
                weights.use { w ->
                    negLogSig.use { nls ->
                        w.mul(nls).use { weightedNeg ->
                            weightedNeg.sum(intArrayOf(1)).use { summed ->
                                summed.neg()
                            }
                        }
                    }
                }
            val rankLoss = posLoss.use { p -> negLoss.use { n -> p.add(n) } }

            val posConsistency = computeConsistencyLoss(posParts)
            val negConsistency = computeConsistencyLoss(negParts)
            val consistency = posConsistency.use { p -> negConsistency.use { n -> p.add(n) } }

            val regLoss = block.l2RegLoss(parameterStore, device, training)
            val loss =
                rankLoss.use { r ->
                    consistency.use { c ->
                        regLoss.use { reg ->
                            c.mul(HYPERCOMPLEX_LAMBDA1).use { cScaled ->
                                reg.mul(HYPERCOMPLEX_LAMBDA2).use { regScaled ->
                                    r.add(cScaled).add(regScaled)
                                }
                            }
                        }
                    }
                }
            return HyperLossResult(
                loss = loss,
                posScores = posScores,
                negScores = negScores,
            )
        } finally {
            posParts.hyperbolic.close()
            posParts.complex.close()
            posParts.euclidean.close()
            negParts.hyperbolic.close()
            negParts.complex.close()
            negParts.euclidean.close()
        }
    }

    private fun computeConsistencyLoss(parts: HyperComplEx.ScoreParts): NDArray {
        val diffHC = parts.hyperbolic.sub(parts.complex)
        val diffHE = parts.hyperbolic.sub(parts.euclidean)
        val diffCE = parts.complex.sub(parts.euclidean)
        return try {
            diffHC.pow(2).use { hc2 ->
                diffHE.pow(2).use { he2 ->
                    diffCE.pow(2).use { ce2 ->
                        hc2.add(he2).use { partial ->
                            partial.add(ce2)
                        }
                    }
                }
            }
        } finally {
            diffHC.close()
            diffHE.close()
            diffCE.close()
        }
    }

    /**
     * Computes a Matryoshka-weighted loss by aggregating per-dimension scores from a RotatE or QuatE block.
     *
     * For each configured Matryoshka dimension that is valid for the block's entity embedding size, this function
     * computes positive and negative scores at that dimension, derives a per-dimension loss via
     * computeLossFromScores(...), multiplies it by the corresponding weight, and returns the sum across dimensions.
     *
     * @param block The model block used for scoring; must be a RotatE or QuatE instance.
     * @param sample NDArray of positive triples (shape [batchSize, 3]).
     * @param negativeSample NDArray of negative triples (shape [batchSize * numNegatives, 3]).
     * @param numNegatives Number of negative samples per positive triple.
     * @param useSimilarityLoss If true, compute similarity-based loss for each dimension; otherwise use hinge loss.
     * @param useSelfAdversarial If true and using similarity loss, apply self-adversarial weighting to negatives.
     * @return An NDArray containing the aggregated, weighted Matryoshka loss across all valid dimensions.
     *
     * @throws IllegalStateException if the provided block is not supported for Matryoshka scoring.
     * @throws IllegalArgumentException if dims and weights lengths differ or no valid Matryoshka dimension exists
     *                                  for the block's embedding size.
     */
    fun computeMatryoshkaLoss(
        block: Any,
        sample: NDArray,
        negativeSample: NDArray,
        numNegatives: Int,
        useSimilarityLoss: Boolean,
        useSelfAdversarial: Boolean,
        higherIsBetter: Boolean,
    ): NDArray {
        val device = sample.device
        // Create a local ParameterStore since Trainer no longer exposes it publicly.
        val parameterStore = ParameterStore(trainer.manager, true)
        val config =
            when (block) {
                is RotatE ->
                    MatryoshkaConfig(
                        MATRYOSHKA_ROTATE_DIMS,
                        MATRYOSHKA_ROTATE_WEIGHTS,
                        block.getEntities(parameterStore, device, true),
                        block.getEdges(parameterStore, device, true),
                    )
                is QuatE ->
                    MatryoshkaConfig(
                        MATRYOSHKA_QUATE_DIMS,
                        MATRYOSHKA_QUATE_WEIGHTS,
                        block.getEntities(parameterStore, device, true),
                        block.getEdges(parameterStore, device, true),
                    )
                else -> error("Matryoshka loss used with unsupported block.")
            }
        val dims = config.dims
        val weights = config.weights
        val entities = config.entities
        val edges = config.edges
        val scorer: (NDArray, Long) -> NDArray =
            when (block) {
                is RotatE -> { input, dim -> block.score(input, entities, edges, dim) }
                is QuatE -> { input, dim -> block.score(input, entities, edges, dim) }
                else -> error("Matryoshka loss used with unsupported block.")
            }
        require(dims.size == weights.size) { "Matryoshka dims and weights must have the same length." }

        val embDim = entities.shape[1]
        val usable = ArrayList<Pair<Long, Float>>(dims.size)
        for (i in dims.indices) {
            val d = dims[i]
            if (d > 0 && d <= embDim) {
                usable.add(d to weights[i])
            }
        }
        require(usable.isNotEmpty()) { "No valid Matryoshka dims for embedding size $embDim." }

        var total: NDArray? = null
        try {
            for ((dim, weight) in usable) {
                val posScores = scorer(sample, dim)
                val negScores = scorer(negativeSample, dim)
                val negReshaped = negScores.reshape(posScores.shape[0], numNegatives.toLong())
                val loss =
                    computeLossFromScores(
                        posScores,
                        negReshaped,
                        useSimilarityLoss,
                        useSelfAdversarial,
                        higherIsBetter,
                    )
                val weighted = loss.mul(weight)
                loss.close()
                negReshaped.close()
                posScores.close()
                negScores.close()
                total =
                    if (total == null) {
                        weighted
                    } else {
                        val sum = total.add(weighted)
                        total.close()
                        weighted.close()
                        sum
                    }
            }
            val result = requireNotNull(total) { "Matryoshka loss total is null." }
            total = null
            return result
        } finally {
            total?.close()
            entities.close()
            edges.close()
        }
    }

    private class MatryoshkaConfig(
        val dims: LongArray,
        val weights: FloatArray,
        val entities: NDArray,
        val edges: NDArray,
    )
}

package jp.live.ugai.tugraph

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.nn.Parameter
import ai.djl.training.ParameterStore
import ai.djl.training.initializer.UniformInitializer
import ai.djl.util.PairList
import kotlin.math.sqrt

/**
 * A class representing the QuatE model.
 *
 * @property numEnt The number of entities.
 * @property numEdge The number of edges.
 * @property dim The dimensionality of the embeddings (per quaternion component).
 */
class QuatE(
    /** Number of entities in the graph. */
    val numEnt: Long,
    /** Number of relations in the graph. */
    val numEdge: Long,
    /** Embedding dimensionality per quaternion component. */
    val dim: Long,
) : AbstractBlock() {
    private val entities: Parameter
    private val edges: Parameter
    private val headIndex = NDIndex(":, 0")
    private val relationIndex = NDIndex(":, 1")
    private val tailIndex = NDIndex(":, 2")
    private val rIndex = NDIndex(":, 0:$dim")
    private val iIndex = NDIndex(":, $dim:${dim * 2}")
    private val jIndex = NDIndex(":, ${dim * 2}:${dim * 3}")
    private val kIndex = NDIndex(":, ${dim * 3}:")
    private val sumAxis = intArrayOf(1)

    init {
        val bound = 6.0f / sqrt(dim.toDouble()).toFloat()
        setInitializer(UniformInitializer(bound), Parameter.Type.WEIGHT)
        entities =
            addParameter(
                Parameter.builder()
                    .setName("entities")
                    .setType(Parameter.Type.WEIGHT)
                    .optShape(Shape(numEnt, dim * 4))
                    .build(),
            )
        edges =
            addParameter(
                Parameter.builder()
                    .setName("edges")
                    .setType(Parameter.Type.WEIGHT)
                    .optShape(Shape(numEdge, dim * 4))
                    .build(),
            )
    }

    /**
     * Computes the forward pass of the QuatE model.
     *
     * @param parameterStore The parameter store.
     * @param inputs The input NDList.
     * @param training Whether the model is in training mode.
     * @param params Additional parameters.
     * @return The output NDList.
     */
    @Override
    override fun forwardInternal(
        parameterStore: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?,
    ): NDList {
        val positive = inputs[0]
        val device = positive.device
        val entitiesArr = parameterStore.getValue(entities, device, training)
        val edgesArr = parameterStore.getValue(edges, device, training)
        val ret = NDList(model(positive, entitiesArr, edgesArr))
        if (inputs.size > 1) {
            ret.add(model(inputs[1], entitiesArr, edgesArr))
        }
        return ret
    }

    /**
     * Compute QuatE scores for a batch of triples using the provided entity and relation embeddings.
     *
     * Scores are the inner product between the Hamilton product (h ⊗ r) and t (higher is better).
     *
     * @param input NDArray of triples where each row is (head, relation, tail) indices.
     * @param entities NDArray of entity embeddings with quaternion components concatenated.
     * @param edges NDArray of relation embeddings with quaternion components concatenated.
     * @return An NDArray of shape (numTriples) containing the QuatE score for each triple.
     */
    fun model(
        input: NDArray,
        entities: NDArray,
        edges: NDArray,
    ): NDArray {
        val numTriples = input.size() / TRIPLE
        var triples: NDArray? = null
        var headIds: NDArray? = null
        var relationIds: NDArray? = null
        var tailIds: NDArray? = null
        var heads: NDArray? = null
        var relations: NDArray? = null
        var tails: NDArray? = null
        var hR: NDArray? = null
        var hI: NDArray? = null
        var hJ: NDArray? = null
        var hK: NDArray? = null
        var rR: NDArray? = null
        var rI: NDArray? = null
        var rJ: NDArray? = null
        var rK: NDArray? = null
        var tR: NDArray? = null
        var tI: NDArray? = null
        var tJ: NDArray? = null
        var tK: NDArray? = null
        var hrR: NDArray? = null
        var hrI: NDArray? = null
        var hrJ: NDArray? = null
        var hrK: NDArray? = null
        var dotR: NDArray? = null
        var dotI: NDArray? = null
        var dotJ: NDArray? = null
        var dotK: NDArray? = null
        var sum: NDArray? = null
        try {
            triples = input.reshape(numTriples, TRIPLE)
            headIds = triples.get(headIndex)
            relationIds = triples.get(relationIndex)
            tailIds = triples.get(tailIndex)

            heads = entities.get(headIds)
            relations = edges.get(relationIds)
            tails = entities.get(tailIds)

            hR = heads.get(rIndex)
            hI = heads.get(iIndex)
            hJ = heads.get(jIndex)
            hK = heads.get(kIndex)
            rR = relations.get(rIndex)
            rI = relations.get(iIndex)
            rJ = relations.get(jIndex)
            rK = relations.get(kIndex)
            tR = tails.get(rIndex)
            tI = tails.get(iIndex)
            tJ = tails.get(jIndex)
            tK = tails.get(kIndex)

            // Hamilton product h ⊗ r
            hrR = hR.mul(rR).sub(hI.mul(rI)).sub(hJ.mul(rJ)).sub(hK.mul(rK))
            hrI = hR.mul(rI).add(hI.mul(rR)).add(hJ.mul(rK)).sub(hK.mul(rJ))
            hrJ = hR.mul(rJ).sub(hI.mul(rK)).add(hJ.mul(rR)).add(hK.mul(rI))
            hrK = hR.mul(rK).add(hI.mul(rJ)).sub(hJ.mul(rI)).add(hK.mul(rR))

            dotR = hrR.mul(tR)
            dotI = hrI.mul(tI)
            dotJ = hrJ.mul(tJ)
            dotK = hrK.mul(tK)
            sum = dotR.add(dotI).add(dotJ).add(dotK)
            return sum.sum(sumAxis)
        } finally {
            sum?.close()
            dotK?.close()
            dotJ?.close()
            dotI?.close()
            dotR?.close()
            hrK?.close()
            hrJ?.close()
            hrI?.close()
            hrR?.close()
            tK?.close()
            tJ?.close()
            tI?.close()
            tR?.close()
            rK?.close()
            rJ?.close()
            rI?.close()
            rR?.close()
            hK?.close()
            hJ?.close()
            hI?.close()
            hR?.close()
            tails?.close()
            relations?.close()
            heads?.close()
            tailIds?.close()
            relationIds?.close()
            headIds?.close()
            triples?.close()
        }
    }

    /**
     * Computes output shapes for the provided input shapes.
     *
     * @param inputs Input shapes for the block.
     * @return Output shapes for the block.
     */
    @Override
    override fun getOutputShapes(inputs: Array<Shape>): Array<Shape> {
        val numTriples = inputs[0].size() / TRIPLE
        val outShape = Shape(numTriples)
        return if (inputs.size > 1) {
            arrayOf(outShape, outShape)
        } else {
            arrayOf(outShape)
        }
    }

    /**
     * Retrieve the entities embeddings array.
     *
     * @return The entities embeddings NDArray with shape (numEnt, dim * 4).
     */
    fun getEntities(): NDArray {
        return getParameters().get("entities").array
    }

    /**
     * Gets relation (edge) embeddings.
     *
     * @return NDArray of shape (numEdge, dim * 4) containing quaternion relation embeddings.
     */
    fun getEdges(): NDArray {
        return getParameters().get("edges").array
    }
}

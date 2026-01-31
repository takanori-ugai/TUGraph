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
 * A class representing the RotatE model.
 *
 * @property numEnt The number of entities.
 * @property numEdge The number of edges.
 * @property dim The dimensionality of the embeddings (per real/imag component).
 */
class RotatE(
    /** Number of entities in the graph. */
    val numEnt: Long,
    /** Number of relations in the graph. */
    val numEdge: Long,
    /** Embedding dimensionality per complex component. */
    val dim: Long,
) : AbstractBlock() {
    private val entities: Parameter
    private val edges: Parameter
    private val headIndex = NDIndex(":, 0")
    private val relationIndex = NDIndex(":, 1")
    private val tailIndex = NDIndex(":, 2")
    private val realIndex = NDIndex(":, 0:$dim")
    private val imagIndex = NDIndex(":, $dim:")
    private val sumAxis = intArrayOf(1)

    init {
        val bound = 6.0f / sqrt(dim.toDouble()).toFloat()
        setInitializer(UniformInitializer(bound), Parameter.Type.WEIGHT)
        entities =
            addParameter(
                Parameter.builder()
                    .setName("entities")
                    .setType(Parameter.Type.WEIGHT)
                    .optShape(Shape(numEnt, dim * 2))
                    .build(),
            )
        edges =
            addParameter(
                Parameter.builder()
                    .setName("edges")
                    .setType(Parameter.Type.WEIGHT)
                    .optShape(Shape(numEdge, dim))
                    .build(),
            )
    }

    /**
     * Computes the forward pass of the RotatE model.
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
     * Compute RotatE scores for a batch of triples using the provided entity and relation embeddings.
     *
     * Scores are L1 distances between rotated head embeddings and tail embeddings (lower is better).
     *
     * @param input NDArray of triples where each row is (head, relation, tail) indices.
     * @param entities NDArray of entity embeddings with real and imaginary components concatenated.
     * @param edges NDArray of relation phase embeddings (radians).
     * @return An NDArray of shape (numTriples) containing the RotatE score for each triple.
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
        var hRe: NDArray? = null
        var hIm: NDArray? = null
        var tRe: NDArray? = null
        var tIm: NDArray? = null
        var rPhase: NDArray? = null
        var rCos: NDArray? = null
        var rSin: NDArray? = null
        var rotRe: NDArray? = null
        var rotIm: NDArray? = null
        var diffRe: NDArray? = null
        var diffIm: NDArray? = null
        var absRe: NDArray? = null
        var absIm: NDArray? = null
        var sum: NDArray? = null
        try {
            triples = input.reshape(numTriples, TRIPLE)
            headIds = triples.get(headIndex)
            relationIds = triples.get(relationIndex)
            tailIds = triples.get(tailIndex)

            heads = entities.get(headIds)
            relations = edges.get(relationIds)
            tails = entities.get(tailIds)

            hRe = heads.get(realIndex)
            hIm = heads.get(imagIndex)
            tRe = tails.get(realIndex)
            tIm = tails.get(imagIndex)

            rPhase = relations
            rCos = rPhase.cos()
            rSin = rPhase.sin()
            rotRe = hRe.mul(rCos).sub(hIm.mul(rSin))
            rotIm = hRe.mul(rSin).add(hIm.mul(rCos))
            diffRe = rotRe.sub(tRe)
            diffIm = rotIm.sub(tIm)
            absRe = diffRe.abs()
            absIm = diffIm.abs()
            sum = absRe.add(absIm)
            return sum.sum(sumAxis)
        } finally {
            sum?.close()
            absIm?.close()
            absRe?.close()
            diffIm?.close()
            diffRe?.close()
            rotIm?.close()
            rotRe?.close()
            rSin?.close()
            rCos?.close()
            tIm?.close()
            tRe?.close()
            hIm?.close()
            hRe?.close()
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
     * @return The entities embeddings NDArray with shape (numEnt, dim * 2).
     */
    fun getEntities(): NDArray {
        return getParameters().get("entities").array
    }

    /**
     * Gets relation (edge) embeddings.
     *
     * @return NDArray of shape (numEdge, dim) containing relation phases.
     */
    fun getEdges(): NDArray {
        return getParameters().get("edges").array
    }
}

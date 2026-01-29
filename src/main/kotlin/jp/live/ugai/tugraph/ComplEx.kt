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
 * A class representing the ComplEx model.
 *
 * @property numEnt The number of entities.
 * @property numEdge The number of edges.
 * @property dim The dimensionality of the embeddings (per real/imag component).
 */
class ComplEx(
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
                    .optShape(Shape(numEdge, dim * 2))
                    .build(),
            )
    }

    /**
     * Computes the forward pass of the ComplEx model.
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
     * Applies the ComplEx scoring function.
     *
     * @param input The input NDArray.
     * @param entities The entities NDArray.
     * @param edges The edges NDArray.
     * @return The output NDArray.
     */
    fun model(
        input: NDArray,
        entities: NDArray,
        edges: NDArray,
    ): NDArray {
        val numTriples = input.size() / TRIPLE
        val triples = input.reshape(numTriples, TRIPLE)

        val headIds = triples.get(headIndex)
        val relationIds = triples.get(relationIndex)
        val tailIds = triples.get(tailIndex)

        val heads = entities.get(headIds)
        val relations = edges.get(relationIds)
        val tails = entities.get(tailIds)

        val hRe = heads.get(realIndex)
        val hIm = heads.get(imagIndex)
        val rRe = relations.get(realIndex)
        val rIm = relations.get(imagIndex)
        val tRe = tails.get(realIndex)
        val tIm = tails.get(imagIndex)

        // ComplEx: Re(<h, r, conj(t)>)
        val term1 = hRe.mul(rRe).sub(hIm.mul(rIm))
        val term2 = hRe.mul(rIm).add(hIm.mul(rRe))
        return term1.mul(tRe).add(term2.mul(tIm)).sum(sumAxis)
    }

    @Override
    /**
     * Computes output shapes for the provided input shapes.
     *
     * @param inputs Input shapes for the block.
     * @return Output shapes for the block.
     */
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
     * Returns the entities NDArray.
     *
     * @return The entities NDArray.
     */
    fun getEntities(): NDArray {
        return getParameters().get("entities").array
    }

    /**
     * Returns the edges NDArray.
     *
     * @return The edges NDArray.
     */
    fun getEdges(): NDArray {
        return getParameters().get("edges").array
    }
}

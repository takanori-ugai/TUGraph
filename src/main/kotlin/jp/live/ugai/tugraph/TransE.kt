package jp.live.ugai.tugraph

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.nn.Parameter
import ai.djl.training.ParameterStore
import ai.djl.training.initializer.UniformInitializer
import ai.djl.util.PairList
import kotlin.math.sqrt

/**
 * A class representing the TransE model.
 *
 * @property numEnt The number of entities.
 * @property numEdge The number of edges.
 * @property dim The dimensionality of the embeddings.
 */
class TransE(val numEnt: Long, val numEdge: Long, val dim: Long) : AbstractBlock() {
    private val entities: Parameter
    private val edges: Parameter

    init {
        val bound = 6.0f / sqrt(dim.toDouble()).toFloat()
        setInitializer(UniformInitializer(bound), Parameter.Type.WEIGHT)
        entities =
            addParameter(
                Parameter.builder()
                    .setName("entities")
                    .setType(Parameter.Type.WEIGHT)
                    .optShape(Shape(numEnt, dim))
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
     * Computes the forward pass of the TransE model.
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
        // Since we added the parameter, we can now access it from the parameter store
        val entitiesArr = parameterStore.getValue(entities, device, training)
        val edgesArr = parameterStore.getValue(edges, device, training)
        val ret = NDList(model(positive, entitiesArr, edgesArr))
        if (inputs.size > 1) {
            ret.add(model(inputs[1], entitiesArr, edgesArr))
        }
        return ret
    }

    /**
     * Applies the linear transformation of the TransE model.
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

        // Gather embeddings for head, relation, and tail entities
        val heads = entities.get(triples.get(":, 0"))
        val relations = edges.get(triples.get(":, 1"))
        val tails = entities.get(triples.get(":, 2"))

        // TransE energy: d(h + r, t) with L1 distance
        return heads.add(relations).sub(tails).abs().sum(intArrayOf(1))
    }

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

    @Override
    override fun initialize(
        manager: ai.djl.ndarray.NDManager,
        dataType: ai.djl.ndarray.types.DataType,
        vararg inputShapes: Shape,
    ) {
        super.initialize(manager, dataType, *inputShapes)
        val rel = getParameters().valueAt(1).array
        val norm = rel.norm(intArrayOf(1), true)
        val safe = norm.maximum(1.0e-12f)
        rel.divi(safe)
    }

    /**
     * Returns the entities NDArray.
     *
     * @return The entities NDArray.
     */
    fun getEntities(): NDArray {
        return getParameters().valueAt(0).array
    }

    /**
     * Returns the edges NDArray.
     *
     * @return The edges NDArray.
     */
    fun getEdges(): NDArray {
        return getParameters().valueAt(1).array
    }

    /**
     * Normalizes the embeddings.
     */
    fun normalize() {
        val ent = getParameters().valueAt(0).array
        val norm = ent.norm(intArrayOf(1), true)
        val safe = norm.maximum(1.0e-12f)
        ent.divi(safe)
    }
}

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
 * A class representing the TransE model.
 *
 * @property numEnt The number of entities.
 * @property numEdge The number of edges.
 * @property dim The dimensionality of the embeddings.
 */
class TransE(
    /** Number of entities in the graph. */
    val numEnt: Long,
    /** Number of relations in the graph. */
    val numEdge: Long,
    /** Embedding dimensionality. */
    val dim: Long,
) : AbstractBlock() {
    private val entities: Parameter
    private val edges: Parameter
    private val headIndex = NDIndex(":, 0")
    private val relationIndex = NDIndex(":, 1")
    private val tailIndex = NDIndex(":, 2")
    private val sumAxis = intArrayOf(1)

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
        var triples: NDArray? = null
        var headIds: NDArray? = null
        var relationIds: NDArray? = null
        var tailIds: NDArray? = null
        var heads: NDArray? = null
        var tails: NDArray? = null
        var relations: NDArray? = null
        var sum: NDArray? = null
        var diff: NDArray? = null
        var abs: NDArray? = null
        try {
            triples = input.reshape(numTriples, TRIPLE)

            // Gather embeddings for head, relation, and tail entities
            headIds = triples.get(headIndex)
            relationIds = triples.get(relationIndex)
            tailIds = triples.get(tailIndex)
            heads = entities.get(headIds)
            tails = entities.get(tailIds)
            relations = edges.get(relationIds)

            // TransE energy: d(h + r, t) with L1 distance
            sum = heads.add(relations)
            diff = sum.sub(tails)
            abs = diff.abs()
            return abs.sum(sumAxis)
        } finally {
            abs?.close()
            diff?.close()
            sum?.close()
            relations?.close()
            tails?.close()
            heads?.close()
            tailIds?.close()
            relationIds?.close()
            headIds?.close()
            triples?.close()
        }
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

    @Override
    /**
     * Initializes parameters and normalizes relation embeddings.
     *
     * @param manager NDManager used for initialization.
     * @param dataType Data type for parameters.
     * @param inputShapes Input shapes for initialization.
     */
    override fun initialize(
        manager: ai.djl.ndarray.NDManager,
        dataType: ai.djl.ndarray.types.DataType,
        vararg inputShapes: Shape,
    ) {
        super.initialize(manager, dataType, *inputShapes)
        val rel = getParameters().valueAt(1).array
        val norm = rel.norm(sumAxis, true)
        val safe = norm.maximum(1.0e-12f)
        try {
            rel.divi(safe)
        } finally {
            safe.close()
            norm.close()
        }
    }

    /**
     * Accesses the tensor of entity embeddings.
     *
     * @return The NDArray containing entity embeddings (shape: [numEnt, dim]).
     */
    fun getEntities(): NDArray {
        return getParameters().get("entities").array
    }

    /**
     * The relation (edge) embedding array used by the model.
     *
     * @return The NDArray containing relation/edge embeddings (shape: [numEdge, dim]).
     */
    fun getEdges(): NDArray {
        return getParameters().get("edges").array
    }

    /**
     * Normalizes the embeddings.
     */
    fun normalize() {
        val ent = getParameters().valueAt(0).array
        val norm = ent.norm(sumAxis, true)
        val safe = norm.maximum(1.0e-12f)
        try {
            ent.divi(safe)
        } finally {
            safe.close()
            norm.close()
        }
    }
}

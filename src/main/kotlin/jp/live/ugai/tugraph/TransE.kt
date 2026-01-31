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
     * Computes TransE energy scores d(h + r, t) using L1 distance for the provided triples.
     *
     * @param input A 1-D or 2-D NDArray of triple indices (flattened or shaped) where each triple is
     *     (head, relation, tail).
     * @param entities Entity embedding matrix with shape [numEnt, dim].
     * @param edges Relation embedding matrix with shape [numEdge, dim].
     * @return An NDArray of shape [numTriples] containing the L1 energy for each triple.
     */
    fun model(
        input: NDArray,
        entities: NDArray,
        edges: NDArray,
    ): NDArray {
        val numTriples = input.size() / TRIPLE
        val parent = input.manager
        return parent.newSubManager().use { sm ->
            val triples = input.reshape(numTriples, TRIPLE).also { it.attach(sm) }

            // Gather embeddings for head, relation, and tail entities
            val headIds = triples.get(headIndex).also { it.attach(sm) }
            val relationIds = triples.get(relationIndex).also { it.attach(sm) }
            val tailIds = triples.get(tailIndex).also { it.attach(sm) }
            val heads = entities.get(headIds).also { it.attach(sm) }
            val tails = entities.get(tailIds).also { it.attach(sm) }
            val relations = edges.get(relationIds).also { it.attach(sm) }

            // TransE energy: d(h + r, t) with L1 distance
            val sum = heads.add(relations).also { it.attach(sm) }
            val diff = sum.sub(tails).also { it.attach(sm) }
            val abs = diff.abs().also { it.attach(sm) }
            val result = abs.sum(sumAxis).also { it.attach(parent) }
            result
        }
    }

    /**
     * Computes the block's output shapes from the given input shapes.
     *
     * @param inputs Input shapes; inputs[0].size() is interpreted as TRIPLE * numTriples and must be
     *     divisible by TRIPLE.
     * @return An array containing one Shape equal to [numTriples], or two identical Shapes when a second
     *     input is provided.
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
     * Initializes model parameters and normalizes relation embeddings to unit L2 norm per embedding,
     * stabilizing the division with a minimum norm of 1e-12.
     *
     * @param manager NDManager used for initialization.
     * @param dataType Data type for parameters.
     * @param inputShapes Input shapes for initialization.
     */
    @Override
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
     * Normalize entity embeddings in-place so each embedding has L2 norm equal to 1 along the embedding axis.
     *
     * Norms smaller than 1e-12 are clamped to 1e-12 to avoid division by zero.
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

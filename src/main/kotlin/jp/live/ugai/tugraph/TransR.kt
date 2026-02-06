package jp.live.ugai.tugraph

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.nn.Parameter
import ai.djl.training.ParameterStore
import ai.djl.training.initializer.UniformInitializer
import ai.djl.util.PairList
import kotlin.math.sqrt

/**
 * A class representing the TransR model.
 *
 * @property numEnt The number of entities.
 * @property numEdge The number of edges.
 * @property dim The dimensionality of the embeddings.
 */
class TransR(
    /** Number of entities in the graph. */
    val numEnt: Long,
    /** Number of relations in the graph. */
    val numEdge: Long,
    /** Entity embedding dimensionality. */
    val entDim: Long,
    /** Relation embedding dimensionality. */
    val relDim: Long = entDim,
) : AbstractBlock() {
    private val entities: Parameter
    private val edges: Parameter
    private val matrix: Parameter
    private val headIndex = NDIndex(":, 0")
    private val relationIndex = NDIndex(":, 1")
    private val tailIndex = NDIndex(":, 2")

    init {
        val bound = 6.0f / sqrt(entDim.toDouble()).toFloat()
        setInitializer(UniformInitializer(bound), Parameter.Type.WEIGHT)
        entities =
            addParameter(
                Parameter
                    .builder()
                    .setName("entities")
                    .setType(Parameter.Type.WEIGHT)
                    .optShape(Shape(numEnt, entDim))
                    .build(),
            )
        edges =
            addParameter(
                Parameter
                    .builder()
                    .setName("edges")
                    .setType(Parameter.Type.WEIGHT)
                    .optShape(Shape(numEdge, relDim))
                    .build(),
            )
        matrix =
            addParameter(
                Parameter
                    .builder()
                    .setName("matrix")
                    .setType(Parameter.Type.WEIGHT)
                    .optShape(Shape(numEdge, relDim, entDim))
                    .build(),
            )
    }

    /**
     * Computes the forward pass of the TransR model.
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
        val matrixArr = parameterStore.getValue(matrix, device, training)
        val ret = NDList(model(positive, entitiesArr, edgesArr, matrixArr))
        if (inputs.size > 1) {
            ret.add(model(inputs[1], entitiesArr, edgesArr, matrixArr))
        }
        return ret
    }

    /**
     * Computes TransR scores for a batch of triples by projecting entity embeddings into relation space
     * and measuring the L1 distance between (projected head + relation) and projected tail for each triple.
     *
     * @param input Flat array of triple IDs arranged as consecutive (head, relation, tail) entries;
     *     length must be a multiple of 3.
     * @param entities Entity embedding matrix with shape (numEntities, entDim).
     * @param edges Relation embedding matrix with shape (numRelations, relDim).
     * @param matrix Relation-specific projection matrices with shape (numRelations, relDim, entDim).
     * @return A 1-D NDArray of length equal to the number of triples containing the L1 score for each
     *     triple.
     */
    fun model(
        input: NDArray,
        entities: NDArray,
        edges: NDArray,
        matrix: NDArray,
    ): NDArray {
        val numTriples = input.size() / TRIPLE
        var triples: NDArray? = null
        var headIds: NDArray? = null
        var relationIds: NDArray? = null
        var tailIds: NDArray? = null
        var heads: NDArray? = null
        var relations: NDArray? = null
        var tails: NDArray? = null
        var matrices: NDArray? = null
        var headsExp: NDArray? = null
        var tailsExp: NDArray? = null
        var headsProjLocalRaw: NDArray? = null
        var tailsProjLocalRaw: NDArray? = null
        var headsProj: NDArray? = null
        var tailsProj: NDArray? = null
        var sum: NDArray? = null
        var diff: NDArray? = null
        var abs: NDArray? = null
        try {
            triples = input.reshape(numTriples, TRIPLE)
            headIds = triples.get(headIndex)
            relationIds = triples.get(relationIndex)
            tailIds = triples.get(tailIndex)

            // Gather embeddings for head, relation, and tail
            heads = entities.get(headIds)
            relations = edges.get(relationIds)
            tails = entities.get(tailIds)

            // Project entities into relation space using relation-specific matrices
            matrices = matrix.get(relationIds)
            headsExp = heads.expandDims(2)
            tailsExp = tails.expandDims(2)
            headsProjLocalRaw = matrices.batchMatMul(headsExp)
            headsProj = headsProjLocalRaw.reshape(Shape(numTriples, relDim))
            tailsProjLocalRaw = matrices.batchMatMul(tailsExp)
            tailsProj = tailsProjLocalRaw.reshape(Shape(numTriples, relDim))

            // TransR energy: d(h_r + r, t_r) with L1 distance
            sum = requireNotNull(headsProj).add(requireNotNull(relations))
            diff = sum.sub(requireNotNull(tailsProj))
            abs = diff.abs()
            return abs.sum(intArrayOf(1))
        } finally {
            abs?.close()
            diff?.close()
            sum?.close()
            tailsProj?.close()
            headsProj?.close()
            tailsProjLocalRaw?.close()
            headsProjLocalRaw?.close()
            tailsExp?.close()
            headsExp?.close()
            matrices?.close()
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
     * Determine the output Shape(s) based on the number of triples encoded in the first input shape.
     *
     * @param inputs Array of input shapes; the first shape's size is expected to be a multiple of the
     *     triple arity.
     * @return An array containing one Shape(numTriples) for a single input, or two identical
     *     Shape(numTriples) entries if a second input is provided.
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
     * Retrieve the entity embedding array registered under the "entities" parameter.
     *
     * @return The NDArray of entity embeddings with shape (numEnt, entDim).
     */
    fun getEntities(): NDArray = getParameters().get("entities").array

    /**
 * Accesses the relation embedding tensor for all relations.
 *
 * @return The relation embeddings as an NDArray with shape (numEdge, relDim).
 */
    fun getEdges(): NDArray = getParameters().get("edges").array

    /**
 * Retrieves the relation projection matrices for all relations.
 *
 * @return NDArray with shape (numEdge, relDim, entDim) containing the per-relation projection matrices.
 */
    fun getMatrix(): NDArray = getParameters().get("matrix").array

    /**
     * Normalize entity and relation embedding rows to unit length.
     *
     * Each row of the entity and relation parameter matrices is divided in place by its L2 norm,
     * with the norm floored to 1e-12 to avoid division by zero.
     */
    fun normalize() {
        val ent = getParameters().get("entities").array
        val rel = getParameters().get("edges").array
        val entNorm = ent.norm(intArrayOf(1), true)
        val entSafe = entNorm.maximum(1.0e-12f)
        val relNorm = rel.norm(intArrayOf(1), true)
        val relSafe = relNorm.maximum(1.0e-12f)
        try {
            ent.divi(entSafe)
            rel.divi(relSafe)
        } finally {
            relSafe.close()
            relNorm.close()
            entSafe.close()
            entNorm.close()
        }
    }

    @Override
    /**
     * Initializes parameters and normalizes embeddings.
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
        normalize()
    }
}
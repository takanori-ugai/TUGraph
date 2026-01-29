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
                Parameter.builder()
                    .setName("entities")
                    .setType(Parameter.Type.WEIGHT)
                    .optShape(Shape(numEnt, entDim))
                    .build(),
            )
        edges =
            addParameter(
                Parameter.builder()
                    .setName("edges")
                    .setType(Parameter.Type.WEIGHT)
                    .optShape(Shape(numEdge, relDim))
                    .build(),
            )
        matrix =
            addParameter(
                Parameter.builder()
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
     * Applies the linear transformation of the TransR model.
     *
     * @param input The input NDArray.
     * @param entities The entities NDArray.
     * @param edges The edges NDArray.
     * @param matrix The matrix NDArray.
     * @return The output NDArray.
     */
    fun model(
        input: NDArray,
        entities: NDArray,
        edges: NDArray,
        matrix: NDArray,
    ): NDArray {
        val numTriples = input.size() / TRIPLE
        val triples = input.reshape(numTriples, TRIPLE)

        // Gather embeddings for head, relation, and tail
        val heads = entities.get(triples.get(headIndex))
        val relations = edges.get(triples.get(relationIndex))
        val tails = entities.get(triples.get(tailIndex))

        // Project entities into relation space using relation-specific matrices
        val matrices = matrix.get(triples.get(relationIndex))
        val headsProj =
            matrices
                .batchMatMul(heads.expandDims(2))
                .reshape(Shape(numTriples, relDim))
        val tailsProj =
            matrices
                .batchMatMul(tails.expandDims(2))
                .reshape(Shape(numTriples, relDim))

        // TransR energy: d(h_r + r, t_r) with L1 distance
        return headsProj.add(relations).sub(tailsProj).abs().sum(intArrayOf(1))
    }

    /**
     * Generates vectorized negative samples for the given triples.
     *
     * This replaces either the head or tail for each triple using a random entity
     * in a vectorized manner. The sampled entity is guaranteed to differ from the
     * original head/tail for the replaced position.
     *
     * @param input Triples NDArray (shape: [batch, 3] or [batch * 3]).
     * @return Negative-sampled triples with the same shape as the reshaped input.
     */
    fun sampleNegatives(input: NDArray): NDArray {
        val numTriples = input.size() / TRIPLE
        val triples = input.reshape(numTriples, TRIPLE)
        val manager = triples.manager

        val heads = triples.get(headIndex)
        val relations = triples.get(relationIndex)
        val tails = triples.get(tailIndex)

        val offsetHead = manager.randomInteger(1, numEnt, Shape(numTriples), DataType.INT64)
        val offsetTail = manager.randomInteger(1, numEnt, Shape(numTriples), DataType.INT64)
        val headSum = heads.add(offsetHead)
        val tailSum = tails.add(offsetTail)
        val headWrap = headSum.gte(numEnt).toType(DataType.INT64, false)
        val tailWrap = tailSum.gte(numEnt).toType(DataType.INT64, false)
        val randHead = headSum.sub(headWrap.mul(numEnt))
        val randTail = tailSum.sub(tailWrap.mul(numEnt))

        val replaceHead = manager.randomInteger(0, 2, Shape(numTriples), DataType.INT64)
        val replaceMask = replaceHead.toType(DataType.INT64, false)
        val keepMask = replaceMask.eq(0).toType(DataType.INT64, false)

        val newHeads = heads.mul(keepMask).add(randHead.mul(replaceMask))
        val newTails = tails.mul(replaceMask).add(randTail.mul(keepMask))

        return newHeads.expandDims(1)
            .concat(relations.expandDims(1), 1)
            .concat(newTails.expandDims(1), 1)
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
        val rel = getParameters().valueAt(1).array
        val entNorm = ent.norm(intArrayOf(1), true).maximum(1.0e-12f)
        val relNorm = rel.norm(intArrayOf(1), true).maximum(1.0e-12f)
        ent.divi(entNorm)
        rel.divi(relNorm)
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

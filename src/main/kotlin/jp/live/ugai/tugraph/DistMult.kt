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
 * A class representing the DistMult model.
 *
 * @property numEnt The number of entities.
 * @property numEdge The number of edges.
 * @property dim The dimensionality of the embeddings.
 */
class DistMult(
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
     * Performs the DistMult forward pass, computing scores for input triples.
     *
     * Computes scores for the first NDList element (positive triples) and, if present, for the second element (negative triples),
     * returning an NDList with one or two NDArrays of per-triple scores.
     *
     * @param inputs NDList where inputs[0] contains triples as rows of (head, relation, tail) and inputs[1] (optional) contains additional triples to score.
     * @return NDList containing one NDArray of scores for each provided input (one or two elements).
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
     * Compute DistMult scores for a batch of triples.
     *
     * The input must contain triples of indices in the order [head, relation, tail] for each triple;
     * it may be shaped (numTriples, 3) or flattened as a multiple of 3. Each returned value is the
     * scalar DistMult score for the corresponding triple.
     *
     * @param input NDArray containing triple indices (head, relation, tail) as described above.
     * @param entities Entity embedding matrix with shape (numEnt, dim); rows map entity IDs to embeddings.
     * @param edges Relation embedding matrix with shape (numEdge, dim); rows map relation IDs to embeddings.
     * @return An NDArray of shape (numTriples) where each entry is the sum over the elementwise product
     *         of the head, relation, and tail embeddings for that triple.
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

        // DistMult: <h, r, t>
        return heads.mul(relations).mul(tails).sum(sumAxis)
    }

    /**
     * Compute the block's output shape(s) from the provided input shape(s) by counting triples in the first input.
     *
     * @param inputs Array of input shapes; the number of triples is computed as inputs[0].size() / 3.
     * @return An array containing a single Shape(numTriples), or two identical Shape(numTriples) entries when two inputs are provided.
     */
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
     * Retrieves the entity embedding matrix.
     *
     * @return The entity embeddings as an NDArray with shape (numEnt, dim).
     */
    fun getEntities(): NDArray {
        return getParameters().get("entities").array
    }

    /**
     * Accesses the NDArray that stores relation (edge) embeddings.
     *
     * @return The relation embeddings NDArray with shape (numEdge, dim).
     */
    fun getEdges(): NDArray {
        return getParameters().get("edges").array
    }
}
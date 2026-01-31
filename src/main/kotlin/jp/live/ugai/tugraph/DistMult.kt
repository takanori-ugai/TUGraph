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
     * Compute DistMult scores for one or two sets of triples.
     *
     * @param inputs NDList where inputs[0] is an NDArray of triples shaped (N, 3) with rows (head, relation, tail);
     *        if present, inputs[1] is a second NDArray of triples to score.
     * @return NDList containing one NDArray of per-triple scores for each provided input (one or two elements).
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
     * Compute scalar DistMult scores for each triple of (head, relation, tail) indices.
     *
     * The input must contain triples of indices in the order [head, relation, tail] for each triple;
     * it may be shaped (numTriples, 3) or flattened as a multiple of 3. Embeddings are looked up
     * by row index from the provided `entities` and `edges` matrices.
     *
     * @param input NDArray containing triple indices in row-major order (shape (numTriples, 3) or flattened as a multiple of 3).
     * @param entities Entity embedding matrix with shape (numEnt, dim); each row is an entity embedding.
     * @param edges Relation embedding matrix with shape (numEdge, dim); each row is a relation embedding.
     * @return An NDArray of shape (numTriples) where each entry is the sum over the elementwise product of the head, relation, and tail embeddings for the corresponding triple.
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
            val headIds = triples.get(headIndex).also { it.attach(sm) }
            val relationIds = triples.get(relationIndex).also { it.attach(sm) }
            val tailIds = triples.get(tailIndex).also { it.attach(sm) }

            val heads = entities.get(headIds).also { it.attach(sm) }
            val relations = edges.get(relationIds).also { it.attach(sm) }
            val tails = entities.get(tailIds).also { it.attach(sm) }

            // DistMult: <h, r, t>
            val prod = heads.mul(relations).also { it.attach(sm) }
            val prod2 = prod.mul(tails).also { it.attach(sm) }
            val result = prod2.sum(sumAxis).also { it.attach(parent) }
            result
        }
    }

    /**
     * Computes the output shape(s) by counting triples in the first input shape.
     *
     * The number of triples is inputs[0].size() / 3. If a second input shape is provided, two identical
     * shapes are returned.
     *
     * @param inputs Input shapes where the first entry represents flattened triples (three values per
     *     triple).
     * @return An array containing one Shape(number of triples), or two identical Shape(number of triples)
     *     when two inputs are provided.
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
     * Get the relation (edge) embeddings.
     *
     * @return The relation embeddings NDArray with shape (numEdge, dim).
     */
    fun getEdges(): NDArray {
        return getParameters().get("edges").array
    }
}
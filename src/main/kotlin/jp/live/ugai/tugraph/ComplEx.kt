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
                Parameter
                    .builder()
                    .setName("entities")
                    .setType(Parameter.Type.WEIGHT)
                    .optShape(Shape(numEnt, dim * 2))
                    .build(),
            )
        edges =
            addParameter(
                Parameter
                    .builder()
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
     * Compute ComplEx scores for a batch of triples using the provided entity and relation embeddings.
     *
     * @param input NDArray of triples where each row is (head, relation, tail) indices.
     * @param entities NDArray of entity embeddings with real and imaginary components concatenated.
     * @param edges NDArray of relation embeddings with real and imaginary components concatenated.
     * @return An NDArray of shape (numTriples) containing the ComplEx score for each triple.
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

            val hRe = heads.get(realIndex).also { it.attach(sm) }
            val hIm = heads.get(imagIndex).also { it.attach(sm) }
            val rRe = relations.get(realIndex).also { it.attach(sm) }
            val rIm = relations.get(imagIndex).also { it.attach(sm) }
            val tRe = tails.get(realIndex).also { it.attach(sm) }
            val tIm = tails.get(imagIndex).also { it.attach(sm) }

            // ComplEx: Re(<h, r, conj(t)>)
            val term1 = hRe.mul(rRe).sub(hIm.mul(rIm)).also { it.attach(sm) }
            val term2 = hRe.mul(rIm).add(hIm.mul(rRe)).also { it.attach(sm) }
            val sum1 = term1.mul(tRe).also { it.attach(sm) }
            val sum2 = term2.mul(tIm).also { it.attach(sm) }
            val sum = sum1.add(sum2).also { it.attach(sm) }
            val result = sum.sum(sumAxis).also { it.attach(parent) }
            result
        }
    }

    /**
     * Infer the output Shape(s) from the supplied input Shapes by computing the number of triples
     * present in the first input.
     *
     * @param inputs Array of input Shapes; the number of triples is computed from inputs[0].size() /
     *     TRIPLE.
     * @return An array containing one Shape (number of triples). If two inputs are provided, returns two
     *     identical Shapes.
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
     * Access the entity embeddings matrix.
     *
     * @return The entity embeddings NDArray with shape (numEnt, dim * 2).
     */
    fun getEntities(): NDArray = getParameters().get("entities").array

    /**
     * Access relation (edge) embeddings.
     *
     * @return NDArray of shape (numEdge, dim * 2) with concatenated real and imaginary components for each relation.
     */
    fun getEdges(): NDArray = getParameters().get("edges").array
}

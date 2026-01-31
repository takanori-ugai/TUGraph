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
        return score(input, entities, edges, dim * 2)
    }

    /**
     * Compute RotatE scores for a batch of triples with a configurable total embedding dimension.
     *
     * The score is the L1 distance ||h ∘ r - t|| where r is a phase rotation (lower is better).
     *
     * @param input NDArray of triples where each row is (head, relation, tail) indices.
     * @param entities NDArray of entity embeddings with real and imaginary components concatenated.
     * @param edges NDArray of relation phase embeddings (radians).
     * @param totalDim Total number of entity embedding dimensions to use (must be even).
     * @return An NDArray of shape (numTriples) containing the RotatE score for each triple.
     */
    fun score(
        input: NDArray,
        entities: NDArray,
        edges: NDArray,
        totalDim: Long,
    ): NDArray {
        require(totalDim % 2L == 0L) { "RotatE totalDim must be even, was $totalDim." }
        val numTriples = input.size() / TRIPLE
        val realDim = totalDim / 2L
        val realIndex = NDIndex(":, 0:$realDim")
        val imagIndex = NDIndex(":, $realDim:$totalDim")
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
            val tRe = tails.get(realIndex).also { it.attach(sm) }
            val tIm = tails.get(imagIndex).also { it.attach(sm) }

            val rPhase = relations.get(NDIndex(":, 0:$realDim")).also { it.attach(sm) }
            val rCos = rPhase.cos().also { it.attach(sm) }
            val rSin = rPhase.sin().also { it.attach(sm) }
            val rotRe = hRe.mul(rCos).sub(hIm.mul(rSin)).also { it.attach(sm) }
            val rotIm = hRe.mul(rSin).add(hIm.mul(rCos)).also { it.attach(sm) }
            val diffRe = rotRe.sub(tRe).also { it.attach(sm) }
            val diffIm = rotIm.sub(tIm).also { it.attach(sm) }
            val absRe = diffRe.abs().also { it.attach(sm) }
            val absIm = diffIm.abs().also { it.attach(sm) }
            val sum = absRe.add(absIm).also { it.attach(sm) }
            val result = sum.sum(sumAxis).also { it.attach(parent) }
            result
        }
    }

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

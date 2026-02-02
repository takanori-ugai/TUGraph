package jp.live.ugai.tugraph

import ai.djl.Device
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
 * A class representing the QuatE model.
 *
 * @property numEnt The number of entities.
 * @property numEdge The number of edges.
 * @property dim The dimensionality of the embeddings (per quaternion component).
 */
class QuatE(
    /** Number of entities in the graph. */
    val numEnt: Long,
    /** Number of relations in the graph. */
    val numEdge: Long,
    /** Embedding dimensionality per quaternion component. */
    val dim: Long,
) : AbstractBlock() {
    private val entities: Parameter
    private val edges: Parameter
    private val headIndex = NDIndex(":, 0")
    private val relationIndex = NDIndex(":, 1")
    private val tailIndex = NDIndex(":, 2")
    private val rIndex = NDIndex(":, 0:$dim")
    private val iIndex = NDIndex(":, $dim:${dim * 2}")
    private val jIndex = NDIndex(":, ${dim * 2}:${dim * 3}")
    private val kIndex = NDIndex(":, ${dim * 3}:")
    private val sumAxis = intArrayOf(1)

    init {
        val bound = 6.0f / sqrt(dim.toDouble()).toFloat()
        setInitializer(UniformInitializer(bound), Parameter.Type.WEIGHT)
        entities =
            addParameter(
                Parameter.builder()
                    .setName("entities")
                    .setType(Parameter.Type.WEIGHT)
                    .optShape(Shape(numEnt, dim * 4))
                    .build(),
            )
        edges =
            addParameter(
                Parameter.builder()
                    .setName("edges")
                    .setType(Parameter.Type.WEIGHT)
                    .optShape(Shape(numEdge, dim * 4))
                    .build(),
            )
    }

    /**
     * Compute QuatE scores for one or two batches of triples and return them as an NDList.
     *
     * @param inputs NDList whose first element is a batch of triples; if a second element is present it is treated as an additional batch to score.
     * @param parameterStore Parameter store used to retrieve model parameters.
     * @param training Whether parameters should be fetched in training mode.
     * @param params Additional parameters (unused by this implementation).
     * @return An NDList containing one NDArray of scores for the first input batch, and a second NDArray if a second input batch was provided.
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
     * Compute QuatE scores for a batch of triples using the provided entity and relation embeddings.
     *
     * Scores are the inner product between the Hamilton product (h ⊗ r) and t (higher is better).
     *
     * @param input NDArray of triples where each row is (head, relation, tail) indices.
     * @param entities NDArray of entity embeddings with quaternion components concatenated.
     * @param edges NDArray of relation embeddings with quaternion components concatenated.
     * @return An NDArray of shape (numTriples) containing the QuatE score for each triple.
     */
    fun model(
        input: NDArray,
        entities: NDArray,
        edges: NDArray,
    ): NDArray {
        return score(input, entities, edges, dim * 4)
    }

    /**
     * Compute per-triple QuatE scores by applying the Hamilton product between head and relation quaternions and
     * taking the inner product with tail quaternions.
     *
     * @param input NDArray of triples where each row is (head, relation, tail) indices.
     * @param entities NDArray of entity embeddings with quaternion components concatenated (R,I,J,K).
     * @param edges NDArray of relation embeddings with quaternion components concatenated (R,I,J,K).
     * @param totalDim Total number of embedding columns to use from `entities`/`edges`; must be divisible by 4.
     * @return An NDArray of shape (numTriples) containing the scalar QuatE score for each triple.
     */
    fun score(
        input: NDArray,
        entities: NDArray,
        edges: NDArray,
        totalDim: Long,
    ): NDArray {
        require(totalDim % 4L == 0L) { "QuatE totalDim must be divisible by 4, was $totalDim." }
        val numTriples = input.size() / TRIPLE
        val compDim = totalDim / 4L
        val rIndex = NDIndex(":, 0:$compDim")
        val iIndex = NDIndex(":, $compDim:${compDim * 2}")
        val jIndex = NDIndex(":, ${compDim * 2}:${compDim * 3}")
        val kIndex = NDIndex(":, ${compDim * 3}:$totalDim")
        val parent = input.manager
        return parent.newSubManager().use { sm ->
            val triples = input.reshape(numTriples, TRIPLE).also { it.attach(sm) }
            val headIds = triples.get(headIndex).also { it.attach(sm) }
            val relationIds = triples.get(relationIndex).also { it.attach(sm) }
            val tailIds = triples.get(tailIndex).also { it.attach(sm) }

            val heads = entities.get(headIds).also { it.attach(sm) }
            val relations = edges.get(relationIds).also { it.attach(sm) }
            val tails = entities.get(tailIds).also { it.attach(sm) }

            val hR = heads.get(rIndex).also { it.attach(sm) }
            val hI = heads.get(iIndex).also { it.attach(sm) }
            val hJ = heads.get(jIndex).also { it.attach(sm) }
            val hK = heads.get(kIndex).also { it.attach(sm) }
            val rR = relations.get(rIndex).also { it.attach(sm) }
            val rI = relations.get(iIndex).also { it.attach(sm) }
            val rJ = relations.get(jIndex).also { it.attach(sm) }
            val rK = relations.get(kIndex).also { it.attach(sm) }
            val tR = tails.get(rIndex).also { it.attach(sm) }
            val tI = tails.get(iIndex).also { it.attach(sm) }
            val tJ = tails.get(jIndex).also { it.attach(sm) }
            val tK = tails.get(kIndex).also { it.attach(sm) }

            // Hamilton product h ⊗ r
            val hrR = hR.mul(rR).sub(hI.mul(rI)).sub(hJ.mul(rJ)).sub(hK.mul(rK)).also { it.attach(sm) }
            val hrI = hR.mul(rI).add(hI.mul(rR)).add(hJ.mul(rK)).sub(hK.mul(rJ)).also { it.attach(sm) }
            val hrJ = hR.mul(rJ).sub(hI.mul(rK)).add(hJ.mul(rR)).add(hK.mul(rI)).also { it.attach(sm) }
            val hrK = hR.mul(rK).add(hI.mul(rJ)).sub(hJ.mul(rI)).add(hK.mul(rR)).also { it.attach(sm) }

            val dotR = hrR.mul(tR).also { it.attach(sm) }
            val dotI = hrI.mul(tI).also { it.attach(sm) }
            val dotJ = hrJ.mul(tJ).also { it.attach(sm) }
            val dotK = hrK.mul(tK).also { it.attach(sm) }
            val sum = dotR.add(dotI).add(dotJ).add(dotK).also { it.attach(sm) }
            val result = sum.sum(sumAxis).also { it.attach(parent) }
            result
        }
    }

    /**
     * Compute the output shape(s) for the given input shapes, producing one output shape per input.
     *
     * The output shape for each input is a one-dimensional Shape whose size equals the number of triples
     * represented by the first input (computed as inputs[0].size() / TRIPLE).
     *
     * @param inputs Array of input shapes; the first element represents triples with width TRIPLE.
     * @return An array of output Shape objects — one Shape per input — each equal to the number of triples.
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
     * Retrieve the entities embeddings array.
     *
     * @return The entities embeddings NDArray with shape (numEnt, dim * 4).
     */
    fun getEntities(): NDArray {
        return getParameters().get("entities").array
    }

    /**
     * Retrieve the entities embeddings on the requested device.
     */
    fun getEntities(
        parameterStore: ParameterStore,
        device: Device,
        training: Boolean,
    ): NDArray {
        return parameterStore.getValue(entities, device, training)
    }

    /**
     * Gets relation (edge) embeddings.
     *
     * @return NDArray of shape (numEdge, dim * 4) containing quaternion relation embeddings.
     */
    fun getEdges(): NDArray {
        return getParameters().get("edges").array
    }

    /**
     * Gets relation (edge) embeddings on the requested device.
     */
    fun getEdges(
        parameterStore: ParameterStore,
        device: Device,
        training: Boolean,
    ): NDArray {
        return parameterStore.getValue(edges, device, training)
    }
}

package jp.live.ugai.tugraph

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.nn.Parameter
import ai.djl.training.ParameterStore
import ai.djl.util.PairList

/**
 * A class representing the TransR model.
 *
 * @property numEnt The number of entities.
 * @property numEdge The number of edges.
 * @property dim The dimensionality of the embeddings.
 */
class TransR(val numEnt: Long, val numEdge: Long, val dim: Long) : AbstractBlock() {
    private val entities: Parameter
    private val edges: Parameter
    private val matrix: Parameter
    private val manager = NDManager.newBaseManager()

    init {
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
        matrix =
            addParameter(
                Parameter.builder()
                    .setName("matrix")
                    .setType(Parameter.Type.WEIGHT)
                    .optShape(Shape(dim, dim))
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
        val heads = entities.get(triples.get(":, 0"))
        val relations = edges.get(triples.get(":, 1"))
        val tails = entities.get(triples.get(":, 2"))

        // Project relation embeddings using the relation-specific matrix
        val projectedRelations = relations.matMul(matrix)

        // Compute TransR score: |head + projectedRelation - tail|, then sum over embedding dimension and normalize
        val score = heads.add(projectedRelations).sub(tails).abs().sum(intArrayOf(1)).div(dim)
        return score
    }

    @Override
    override fun getOutputShapes(inputs: Array<Shape>): Array<Shape> {
        return arrayOf<Shape>(Shape(1), Shape(1))
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
        getParameters().forEach {
            it.value.array.norm()
//            it.value.array.divi(it.value.array.pow(2).sum().sqrt())
        }
    }
}

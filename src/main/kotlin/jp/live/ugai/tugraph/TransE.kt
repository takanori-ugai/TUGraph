package jp.live.ugai.tugraph

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.nn.Parameter
import ai.djl.training.ParameterStore
import ai.djl.util.PairList

class TransE(val numEnt: Long, val numEdge: Long, val dim: Long) : AbstractBlock() {

    private val entities: Parameter
    private val edges: Parameter
    private val manager = NDManager.newBaseManager()

    init {
        entities = addParameter(
            Parameter.builder()
                .setName("entities")
                .setType(Parameter.Type.WEIGHT)
                .optShape(Shape(numEnt, dim))
                .build()
        )
        edges = addParameter(
            Parameter.builder()
                .setName("edges")
                .setType(Parameter.Type.WEIGHT)
                .optShape(Shape(numEdge, dim))
                .build()
        )
    }

    @Override
    override fun forwardInternal(
        parameterStore: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?
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

    // Applies linear transformation
    fun model(input: NDArray, entities: NDArray, edges: NDArray): NDArray {
        var v = manager.zeros(Shape(0))
        val inputs = input.reshape(input.size() / 3, 3)
        for (i in 0 until input.size() / 3) {
            val line0 = inputs.get(i).toLongArray()
            v = v.concat(entities.get(line0[0]).add(edges.get(line0[1])).sub(entities.get(line0[2])))
        }
//        val ret = v.reshape(input.size() / 3, dim).pow(2).sum(intArrayOf(1)).sqrt()
        val ret = v.reshape(input.size() / 3, dim).abs().sum(intArrayOf(1)).div(dim)
        return ret
    }

    @Override
    override fun getOutputShapes(inputs: Array<Shape>): Array<Shape> {
        return arrayOf<Shape>(Shape(1), Shape(1))
    }

    fun getEntities(): NDArray {
        return getParameters().valueAt(0).array
    }

    fun getEdges(): NDArray {
        return getParameters().valueAt(1).array
    }

    fun normalize() {
        getParameters().forEach {
            it.value.array.norm()
//            it.value.array.divi(it.value.array.pow(2).sum().sqrt())
        }
    }
}

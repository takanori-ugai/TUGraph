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
        val input = inputs.singletonOrThrow()
        val device = input.device
        // Since we added the parameter, we can now access it from the parameter store
        val entitiesArr = parameterStore.getValue(entities, device, false)
        val edgesArr = parameterStore.getValue(edges, device, false)
        return transE(input, entitiesArr, edgesArr)
    }

    // Applies linear transformation
    fun transE(input: NDArray, entities: NDArray, edges: NDArray): NDList {
        var v = manager.zeros(Shape(0))
        val inputs = input.reshape(input.size() / 3, 3)
        for (i in 0 until input.size() / 3) {
            val line0 = inputs.get(i).toLongArray()
            v = v.concat(entities.get(line0[0]).add(edges.get(line0[1])).sub(entities.get(line0[2])))
        }
        val ret = v.reshape(input.size() / 3, dim).pow(2).sum(intArrayOf(1)).sqrt()
        return NDList(ret)
    }

    @Override
    override fun getOutputShapes(inputs: Array<Shape>): Array<Shape> {
        return arrayOf<Shape>(Shape(1, dim))
    }

    fun getEntities(): NDArray {
        return getParameters().valueAt(0).array
    }

    fun getEdges(): NDArray {
        return getParameters().valueAt(1).array
    }

    fun normalize() {
        getParameters().forEach {
            it.value.array.divi(it.value.array.pow(2).sum().sqrt())
        }
    }
}

package jp.live.ugai.tugraph

import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape

fun main() {
    val rank = mapOf<Int, Float>(
        2 to 0.2f,
        3 to 0.7f,
        4 to 0.3f
    )
//    val comparator : Comparator<<Float> = compareBy { it }
    val lengthOrder = rank.toList().sortedBy { it.second }
    println(lengthOrder.indexOf(Pair(3, rank[3])))
    println(lengthOrder)
}

class Test7
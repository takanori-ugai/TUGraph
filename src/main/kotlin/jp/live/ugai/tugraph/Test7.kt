package jp.live.ugai.tugraph

import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex

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
    val manager = NDManager.newBaseManager()
    val matrix = manager.arange(20).reshape(4, 5)
    println(matrix)
    val mini = matrix.get(NDIndex("{},", manager.create(intArrayOf(0, 2))))
    val mini2 = manager.arange(5).reshape(1, 5)
    println(mini2.transpose())
    println(mini.sum(intArrayOf(0)).sub(mini2))
    println(mini.sum(intArrayOf(0)).transpose().sub(mini2.transpose()).transpose())
    val mini3 = manager.arange(4).reshape(1, 4)
    println(matrix.get(NDIndex("0:-1")))
    println(matrix.transpose().get(NDIndex("0:-1")).concat(mini3).transpose())
    val mini4 = manager.arange(5).reshape(1, 5)
    println(mini4.repeat(0, 2))
}

class Test7

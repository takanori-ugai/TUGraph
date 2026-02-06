package jp.live.ugai.tugraph

import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.Shape
import ai.djl.training.dataset.ArrayDataset

/**
 * Demonstrates NDArray creation, indexing, basic tensor operations, and ArrayDataset assembly using an NDManager, printing intermediate results.
 *
 * Performs a small example workflow: sorts a map of ranks and prints the index and sorted list; creates a 4×5 matrix and prints various slices, reductions, transposes, concatenations, and repeats; builds and prepares an ArrayDataset. All outputs are written to standard output and NDManager resources are released after use.
 */
fun main() {
    val rank =
        mapOf<Int, Float>(
            2 to 0.2f,
            3 to 0.7f,
            4 to 0.3f,
        )
//    val comparator : Comparator<<Float> = compareBy { it }
    val lengthOrder = rank.toList().sortedBy { it.second }
    println(lengthOrder.indexOf(Pair(3, rank[3])))
    println(lengthOrder)
    NDManager.newBaseManager().use { manager ->
        val matrix = manager.arange(20).reshape(4, 5)
        println(matrix)
        val mini = matrix.get(NDIndex("{},", manager.create(intArrayOf(0, 2))))
        val mini2 = manager.arange(5).reshape(1, 5)
        println(mini2.transpose())
        println(mini.sum(intArrayOf(0)).sub(mini2))
        println(
            mini
                .sum(intArrayOf(0))
                .transpose()
                .sub(mini2.transpose())
                .transpose(),
        )
        val mini3 = manager.arange(4).reshape(1, 4)
        println(matrix.get(NDIndex("0:-1")))
        println(
            matrix
                .transpose()
                .get(NDIndex("0:-1"))
                .concat(mini3)
                .transpose(),
        )
        val mini4 = manager.arange(5).reshape(1, 5)
        println(mini4.repeat(0, 2))

        val dataset =
            ArrayDataset
                .Builder()
                .setData(manager.arange(0, 4), manager.arange(5, 9))
                .optLabels(manager.ones(Shape(1)), manager.zeros(Shape(1)))
                .setSampling(20, false)
                .build()
        dataset.prepare()
    }
}

/** Marker class for Test7 example. */
class Test7

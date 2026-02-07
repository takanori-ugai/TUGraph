package jp.live.ugai.tugraph.demo

import ai.djl.Model
import ai.djl.metric.Metrics
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.types.Shape
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.Trainer
import jp.live.ugai.tugraph.TRIPLE

data class TriplesInfo(
    val input: NDArray,
    val inputList: List<LongArray>,
    val numEntities: Long,
    val numEdges: Long,
    val numTriples: Int,
)

fun buildTriplesInfo(input: NDArray): TriplesInfo {
    val totalTriples = input.size() / TRIPLE
    require(totalTriples * TRIPLE == input.size()) {
        "Triples NDArray size must be a multiple of TRIPLE; size=${input.size()}."
    }
    require(totalTriples <= Int.MAX_VALUE.toLong()) {
        "Dataset exceeds Int.MAX_VALUE triples; Long indexing required (numTriples=$totalTriples)."
    }
    val numTriples = totalTriples.toInt()

    val flat = input.toLongArray()
    val inputList = ArrayList<LongArray>(numTriples)
    var idx = 0
    repeat(numTriples) {
        inputList.add(longArrayOf(flat[idx], flat[idx + 1], flat[idx + 2]))
        idx += 3
    }

    val needsReshape = input.shape.dimension() == 1 || input.shape[1] != TRIPLE
    val triples =
        if (needsReshape) {
            input.reshape(totalTriples, TRIPLE)
        } else {
            input
        }
    try {
        val headMax =
            triples.get(":, 0").use { col ->
                col.max().use { it.toLongArray()[0] }
            }
        val tailMax =
            triples.get(":, 2").use { col ->
                col.max().use { it.toLongArray()[0] }
            }
        val relMax =
            triples.get(":, 1").use { col ->
                col.max().use { it.toLongArray()[0] }
            }
        val numEntities = maxOf(headMax, tailMax) + 1
        val numEdges = relMax + 1
        return TriplesInfo(input, inputList, numEntities, numEdges, numTriples)
    } finally {
        if (needsReshape) {
            triples.close()
        }
    }
}

fun newTrainer(
    model: Model,
    config: DefaultTrainingConfig,
    inputShape: Shape,
): Trainer =
    model.newTrainer(config).also {
        it.initialize(inputShape)
        it.metrics = Metrics()
    }

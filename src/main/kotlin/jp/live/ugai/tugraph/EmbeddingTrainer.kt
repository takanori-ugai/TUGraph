package jp.live.ugai.tugraph

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.training.Trainer
import kotlin.random.Random

/**
 * A class for training an embedding model.
 *
 * @property manager The NDManager instance used for creating NDArrays.
 * @property triples The NDArray containing the training triples.
 * @property numOfEntities The number of entities in the training data.
 * @property trainer The Trainer instance for training the model.
 * @property epoch The number of training epochs.
 */
class EmbeddingTrainer(
    private val manager: NDManager,
    private val triples: NDArray,
    private val numOfEntities: Long,
    private val trainer: Trainer,
    private val epoch: Int,
) {
    private val margin = 1.0f

    /** Per-epoch loss values collected during training. */
    val lossList = mutableListOf<Float>()
    private val numOfTriples = triples.shape[0]

    /** Materialized triples used for negative sampling. */
    val inputList = mutableListOf<List<Long>>()

    init {
        for (i in 0 until numOfTriples) {
            inputList.add(triples.get(i).toLongArray().toList())
        }
    }

    /**
     * Trains the embedding model.
     */
    fun training() {
        (1..epoch).forEach {
            var epochLoss = 0f
            for (i in 0 until numOfTriples) {
                val sample = triples.get(i)
                val element = inputList.elementAt(i.toInt())
                val fst = element.elementAt(0)
                val sec = element.elementAt(1)
                val trd = element.elementAt(2)
                val nsample =
                    if (Random.nextBoolean()) {
                        var ran = Random.nextLong(numOfEntities)
                        while (ran == fst || inputList.contains(listOf(fst, sec, ran))) {
                            ran = Random.nextLong(numOfEntities)
                        }
                        manager.create(longArrayOf(fst, sec, ran))
                    } else {
                        var ran = Random.nextLong(numOfEntities)
                        while (ran == trd || inputList.contains(listOf(ran, sec, trd))) {
                            ran = Random.nextLong(numOfEntities)
                        }
                        manager.create(longArrayOf(ran, sec, trd))
                    }
                trainer.newGradientCollector().use { gc ->
                    val f0 = trainer.forward(NDList(sample, nsample))
                    val pos = f0[0]
                    val neg = f0[1]
                    val hinge = pos.sub(neg).add(margin).maximum(0f)
                    epochLoss += hinge.sum().getFloat()
                    gc.backward(hinge.mean())
//                    gc.close()
                }
                trainer.step()
                when (val block = trainer.model.block) {
                    is TransE -> block.normalize()
                    is TransR -> block.normalize()
                }
            }
            val logEvery = maxOf(1, epoch / 500L)
            if (it % logEvery == 0L) {
                println("Epoch $it finished. (L2Loss: ${epochLoss / numOfTriples})")
            }
            lossList.add(epochLoss)
//        trainer.notifyListeners { listener -> listener.onEpoch(trainer) }
        }
        trainer.close()
        manager.close()
    }
}

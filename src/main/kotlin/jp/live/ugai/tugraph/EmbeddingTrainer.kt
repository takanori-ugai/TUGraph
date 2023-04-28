package jp.live.ugai.tugraph

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import ai.djl.training.Trainer
import kotlin.random.Random

class EmbeddingTrainer(
    private val manager: NDManager,
    private val triples: NDArray,
    private val numOfEntities: Long,
    private val trainer: Trainer,
    private val epoch: Int
) {
    val lossList = mutableListOf<Float>()
    private val numOfTriples = triples.shape[0]
    val inputList = mutableListOf<List<Long>>()

    init {
        for (i in 0 until numOfTriples) {
            inputList.add(triples.get(i).toLongArray().toList())
        }
    }

    fun training() {
        (1..epoch).forEach {
            var epochLoss = 0f
            (0 until numOfTriples).forEach {
                val sample = triples.get(it)
                val element = inputList.elementAt(it.toInt())
                val fst = element.elementAt(0)
                val sec = element.elementAt(1)
                val trd = element.elementAt(2)
                val nsample =
                    if (Random.nextInt(1) == 0) {
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
                    val l = trainer.loss.evaluate(NDList(manager.ones(Shape(1))), NDList(f0[1].sub(f0[0])))
                    epochLoss += l.sum().toFloatArray()[0]
                    gc.backward(l)
                    gc.close()
                }
                trainer.step()
//                (trainer.model.block as TransE).normalize()
            }
            if (it % (epoch / 2L) == 0L) {
                println("Epoch $it finished. (L2Loss: ${epochLoss / numOfTriples})")
            }
            lossList.add(epochLoss)
//        trainer.notifyListeners { listener -> listener.onEpoch(trainer) }
        }
        trainer.close()
        manager.close()
    }
}

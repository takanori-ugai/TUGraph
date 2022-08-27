package jp.live.ugai.tugraph

import ai.djl.Model
import ai.djl.metric.Metrics
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Parameter
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.initializer.NormalInitializer
import ai.djl.training.listener.TrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import ai.djl.translate.NoopTranslator
import kotlin.random.Random

fun main() {
    val manager = NDManager.newBaseManager()

    val input = manager.create(triples, Shape(NUM_TRIPLE, TRIPLE))
    val inputList = mutableListOf<List<Long>>()
    (0 until NUM_TRIPLE).forEach {
        inputList.add(input.get(it).toLongArray().toList())
    }
//    println(inputList.contains(listOf<Long>(2,0,1)))
    val labels = manager.create(labelData)

    val transe = TransE(NUM_ENTITIES, NUM_EDGES, DIMENSION).also {
        it.setInitializer(NormalInitializer(), Parameter.Type.WEIGHT)
        it.initialize(manager, DataType.FLOAT32, input.shape)
    }
    val model = Model.newInstance("transe").also {
        it.block = transe
    }
    val predictor = model.newPredictor(NoopTranslator())

    val l2loss = Loss.l2Loss()
    val lrt = Tracker.fixed(LEARNING_RATE)
    val sgd = Optimizer.sgd().setLearningRateTracker(lrt).build()

    val config = DefaultTrainingConfig(l2loss)
        .optOptimizer(sgd) // Optimizer (loss function)
        .optDevices(manager.engine.getDevices(1)) // single GPU
        .addTrainingListeners(*TrainingListener.Defaults.logging()) // Logging

    val trainer = model.newTrainer(config).also {
        it.initialize(Shape(NUM_ENTITIES, TRIPLE))
        it.metrics = Metrics()
    }

    val lossList = mutableListOf<Float>()
    (0..NEPOCH).forEach {
        var epochLoss = 0f
        (0 until NUM_TRIPLE).forEach {
            val X = input.get(it)
//            println(X)
            val y = labels.get(it)
//            println("${element} and ${listOf(fst, sec, ran)}")
//            println(y)
            trainer.newGradientCollector().use { gc ->
                val f0 = trainer.forward(NDList(X))
                val l = f0.singletonOrThrow()
                epochLoss += l.sum().toFloatArray()[0]
//             if(i % 100 == 0) {
//                  println("l = $l")
//             }
                gc.backward(l)
                gc.close()
                trainer.step()
            }
            val element = inputList.elementAt(it.toInt())
            val fst = element.elementAt(0)
            val sec = element.elementAt(1)
            val trd = element.elementAt(2)
            if (Random.nextInt(1) == 0) {
                var ran = Random.nextLong(NUM_ENTITIES)
                while (ran == fst || inputList.contains(listOf(fst, sec, ran))) {
                    ran = Random.nextLong(NUM_ENTITIES)
                }
                trainer.newGradientCollector().use { gc ->
                    val f0 = trainer.forward(NDList(manager.create(longArrayOf(fst, sec, ran))))
                    val l = f0.singletonOrThrow().sub(1f).abs()
                    epochLoss += l.sum().toFloatArray()[0]
//             if(i % 100 == 0) {
//                  println("l = $l")
//             }
                    gc.backward(l)
                    gc.close()
                    trainer.step()
                }
            } else {
                var ran = Random.nextLong(NUM_ENTITIES)
                while (ran == trd || inputList.contains(listOf(ran, sec, trd))) {
                    ran = Random.nextLong(NUM_ENTITIES)
                }
                trainer.newGradientCollector().use { gc ->
                    val f0 = trainer.forward(NDList(manager.create(longArrayOf(ran, sec, trd))))
                    val l = f0.singletonOrThrow().sub(1f).abs()
                    epochLoss += l.sum().toFloatArray()[0]
//             if(i % 100 == 0) {
//                  println("l = $l")
//             }
                    gc.backward(l)
                    gc.close()
                    trainer.step()
                }
            }

        }
//        println("${it} : ${epochLoss/NUM_TRIPLE/2}")
        lossList.add(epochLoss)
    }

    println(transe.getEdges())
    println(transe.getEntities())
//    println(loss)
    println(predictor.predict(NDList(input)).singletonOrThrow())
    val test = manager.create(longArrayOf(1, 1, 2))
    println(predictor.predict(NDList(test)).singletonOrThrow())
    //   println(predictor.predict(NDList(test)).singletonOrThrow().pow(2).sum().sqrt())
}


class Test6

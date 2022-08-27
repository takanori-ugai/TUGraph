package jp.live.ugai.tugraph

import ai.djl.Model
import ai.djl.metric.Metrics
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Parameter
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.initializer.NormalInitializer
import ai.djl.training.listener.TrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import ai.djl.translate.NoopTranslator
import kotlin.random.Random

fun main() {
    val manager = NDManager.newBaseManager()
    val csvReader = CsvToNdarray(manager)

    val input = csvReader.read("data/sample.csv")
    println(input)
    val numOfTriples = input.shape[0]
    val inputList = mutableListOf<List<Long>>()
    (0 until numOfTriples).forEach {
        inputList.add(input.get(it).toLongArray().toList())
    }

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
        (0 until numOfTriples).forEach {
            val X = input.get(it)
            trainer.newGradientCollector().use { gc ->
                val f0 = trainer.forward(NDList(X))
                val l = f0.singletonOrThrow()
                epochLoss += l.toFloatArray()[0]
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
                    epochLoss += l.toFloatArray()[0]
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
                    epochLoss += l.toFloatArray()[0]
                    gc.backward(l)
                    gc.close()
                    trainer.step()
                }
            }
        }
        if (it % (NEPOCH / 10L) == 0L) {
            println("$it : ${epochLoss / NUM_TRIPLE / 2}")
        }
        lossList.add(epochLoss)
    }

    println(transe.getEdges())
    println(transe.getEntities())
//    println(predictor.predict(NDList(input)).singletonOrThrow())
    val test = manager.create(longArrayOf(1, 1, 2))
    println(predictor.predict(NDList(test)).singletonOrThrow())
    inputList.forEach { triple ->
        val rank = mutableMapOf<Long, Float>()
        (0 until NUM_ENTITIES).forEach {
            val t0 = manager.create(longArrayOf(triple[0], triple[1], it))
            rank[it] = predictor.predict(NDList(t0)).singletonOrThrow().toFloatArray()[0]
//            println("${triple}, $it, ${predictor.predict(NDList(t0)).singletonOrThrow()}")
        }
        val lengthOrder = rank.toList().sortedBy { it.second }
        println("$triple, ${lengthOrder.indexOf(Pair(triple[2], rank[triple[2]])) + 1}")
    }

//    println(predictor.predict(NDList(input)).singletonOrThrow())
}

class Test6

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

    val lrt = Tracker.fixed(LEARNING_RATE)
    val sgd = Optimizer.sgd().setLearningRateTracker(lrt).build()

    val config = DefaultTrainingConfig(Loss.l1Loss())
        .optOptimizer(sgd) // Optimizer (loss function)
        .optDevices(manager.engine.getDevices(1)) // single GPU
        .addTrainingListeners(*TrainingListener.Defaults.logging()) // Logging

    val trainer = model.newTrainer(config).also {
        it.initialize(Shape(NUM_ENTITIES, TRIPLE))
        it.metrics = Metrics()
    }

    val eTrainer = EmbeddingTrainer(manager.newSubManager(), input, NUM_ENTITIES, trainer, 10)
    eTrainer.training()
    println(trainer.trainingResult)

    println(transe.getEdges())
    println(transe.getEntities())

    val test = manager.create(longArrayOf(1, 1, 2))
    println(predictor.predict(NDList(test)).singletonOrThrow())

    val result = ResultEval(inputList, manager.newSubManager(), predictor)
    println("Tail")
    result.getTailResult().forEach {
        println("${it.key} : ${it.value}")
    }
    println("Head")
    result.getHeadResult().forEach {
        println("${it.key} : ${it.value}")
    }
    result.close()
}

class Test6

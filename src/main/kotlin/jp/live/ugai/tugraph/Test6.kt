package jp.live.ugai.tugraph

import ai.djl.Model
import ai.djl.metric.Metrics
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.listener.EpochTrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import ai.djl.translate.NoopTranslator

/** Runs TransE training and evaluation on a CSV dataset. */
/**
 * Runs the end-to-end embedding training and evaluation pipeline using DJL within a managed NDManager scope.
 *
 * Loads triples from "data/sample.csv", constructs and initializes a TransE model, configures training
 * (optimizer, devices, and listeners), and performs training with EmbeddingTrainer. After training, prints
 * training results, model edges and entities, performs a sample prediction, and computes head/tail evaluation
 * results; all DJL resources are created and released inside the managed NDManager scope.
 */
fun main() {
    NDManager.newBaseManager().use { manager ->
        val csvReader = CsvToNdarray(manager)
        val input = csvReader.read("data/sample.csv")
        println(input)
        val numOfTriples = input.shape[0]
        val inputList = mutableListOf<LongArray>()
        for (i in 0 until numOfTriples) {
            inputList.add(input.get(i).toLongArray())
        }
        val headMax = input.get(":, 0").max().toLongArray()[0]
        val tailMax = input.get(":, 2").max().toLongArray()[0]
        val relMax = input.get(":, 1").max().toLongArray()[0]
        val numEntities = maxOf(headMax, tailMax) + 1
        val numEdges = relMax + 1
        val transe =
            TransE(numEntities, numEdges, DIMENSION).also {
                it.initialize(manager, DataType.FLOAT16, input.shape)
            }
        val model =
            Model.newInstance("transe").also {
                it.block = transe
            }
        val predictor = model.newPredictor(NoopTranslator())

        val lrt = Tracker.fixed(LEARNING_RATE)
        val sgd = Optimizer.sgd().setLearningRateTracker(lrt).build()

        val config =
            DefaultTrainingConfig(Loss.l1Loss()) // Placeholder loss; EmbeddingTrainer computes its own.
                .optOptimizer(sgd) // Optimizer (loss function)
                .optDevices(manager.engine.getDevices(1)) // single GPU
                .addTrainingListeners(EpochTrainingListener(), HingeLossLoggingListener()) // Hinge loss logging

        val trainer =
            model.newTrainer(config).also {
                it.initialize(input.shape)
                it.metrics = Metrics()
            }

        val eTrainer = EmbeddingTrainer(manager.newSubManager(), input, numEntities, trainer, NEPOCH)
        eTrainer.training()
        input.close()
        println("Training Result")
        println(trainer.trainingResult)

        println("Edge")
        println(transe.getEdges())
        println("Entities")
        println(transe.getEntities())

        val test = manager.create(longArrayOf(1, 1, 2))
        print("Predict (False):")
        println(predictor.predict(NDList(test)).singletonOrThrow().toFloatArray()[0])
        test.close()

        val result = ResultEval(inputList, manager.newSubManager(), predictor, numEntities)
        println("Tail")
        result.getTailResult().forEach {
            println("${it.key} : ${it.value}")
        }
        println("Head")
        result.getHeadResult().forEach {
            println("${it.key} : ${it.value}")
        }
        result.close()
        predictor.close()
    }
}

/** Marker class for Test6 example. */
class Test6

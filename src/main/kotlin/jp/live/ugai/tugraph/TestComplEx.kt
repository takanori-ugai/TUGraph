package jp.live.ugai.tugraph

import ai.djl.Model
import ai.djl.metric.Metrics
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.listener.EpochTrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.tracker.Tracker
import ai.djl.translate.NoopTranslator

/**
 * Orchestrates data loading, model construction, training, evaluation, and a sample prediction within a managed
 * NDManager scope.
 *
 * Reads triple data from "data/sample.csv"; builds and initializes a ComplEx knowledge-graph embedding model;
 * configures training (optimizer, devices, listeners) and runs embedding training for a fixed number of epochs;
 * prints training results and the model's learned entities/edges; runs a single sample prediction; evaluates and
 * prints head and tail ranking results.
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
        val complex =
            ComplEx(numEntities, numEdges, DIMENSION).also {
                it.initialize(manager, DataType.FLOAT32, input.shape)
            }
        val model =
            Model.newInstance("complex").also {
                it.block = complex
            }
        val predictor = model.newPredictor(NoopTranslator())

        val lrt = Tracker.fixed(ADAGRAD_LEARNING_RATE)
        val adagrad = DenseAdagrad.builder().optLearningRateTracker(lrt).build()

        val config =
            DefaultTrainingConfig(Loss.l1Loss()) // Placeholder loss; EmbeddingTrainer computes its own.
                .optOptimizer(adagrad) // Optimizer (loss function)
                .optDevices(manager.engine.getDevices(1)) // single GPU
                .addTrainingListeners(EpochTrainingListener(), HingeLossLoggingListener()) // Hinge loss logging

        val trainer =
            model.newTrainer(config).also {
                it.initialize(input.shape)
                it.metrics = Metrics()
            }

        val eTrainer = EmbeddingTrainer(manager.newSubManager(), input, numEntities, trainer, NEPOCH)
        eTrainer.training()
        println(trainer.trainingResult)

        println(complex.getEdges())
        println(complex.getEntities())

        val test = manager.create(longArrayOf(1, 1, 2))
        println(predictor.predict(NDList(test)).singletonOrThrow())

        val result =
            ResultEval(
                inputList,
                manager.newSubManager(),
                predictor,
                numEntities,
                higherIsBetter = true,
                complEx = complex,
            )
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
}

/** Marker class for TestComplEx example. */
class TestComplEx

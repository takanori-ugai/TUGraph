package jp.live.ugai.tugraph

import ai.djl.Model
import ai.djl.metric.Metrics
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.listener.TrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import ai.djl.translate.NoopTranslator

/**
 * Loads triplet data from CSV, constructs and trains a TransE embedding model, and runs a single prediction.
 *
 * Reads "data/sample.csv" into an NDArray, derives entity and relation counts from the data, initializes
 * a TransE block and DJL model, configures an SGD optimizer and training config, executes embedding training
 * via EmbeddingTrainer, prints the trainer's results, and performs one prediction using the trained model.
 */
fun main() {
    NDManager.newBaseManager().use { manager ->
        val csvReader = CsvToNdarray(manager)

        val input = csvReader.read("data/sample.csv")
        println(input)
        val headMax = input.get(":, 0").max().toLongArray()[0]
        val tailMax = input.get(":, 2").max().toLongArray()[0]
        val relMax = input.get(":, 1").max().toLongArray()[0]
        val numEntities = maxOf(headMax, tailMax) + 1
        val numEdges = relMax + 1

        val transe =
            TransE(numEntities, numEdges, DIMENSION).also {
                it.initialize(manager, DataType.FLOAT32, input.shape)
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
                .apply { TrainingListener.Defaults.logging().forEach { addTrainingListeners(it) } } // Logging

        val trainer =
            model.newTrainer(config).also {
                it.initialize(input.shape)
                it.metrics = Metrics()
            }

        val eTrainer = EmbeddingTrainer(manager.newSubManager(), input, numEntities, trainer, 10)
        eTrainer.training()
        println(trainer.trainingResult)

        val test = manager.create(longArrayOf(1, 1, 2))
        print("Predict (False):")
        println(predictor.predict(ai.djl.ndarray.NDList(test)).singletonOrThrow().toFloatArray()[0])
    }
}

/** Marker class for Test8 example. */
class Test8

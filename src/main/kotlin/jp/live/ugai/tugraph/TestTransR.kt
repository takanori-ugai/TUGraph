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
import jp.live.ugai.tugraph.eval.ResultEvalTransR
import jp.live.ugai.tugraph.train.EmbeddingTrainer

/**
 * Runs an end-to-end example that trains and evaluates a TransR embedding model from CSV data.
 *
 * This function performs the complete workflow for the example:
 * - Reads triples from "data/sample.csv" into an NDArray and converts them to a list of long triples.
 * - Constructs and initializes a TransR model and wraps it in a DJL Model and predictor.
 * - Configures an optimizer and training setup, creates a Trainer, and runs embedding training via EmbeddingTrainer.
 * - Prints training results, learned entity and edge embeddings, and a sample model prediction.
 * - Evaluates head and tail prediction performance using ResultEvalTransR and prints the evaluation results.
 * - Closes all managed resources before exiting.
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
        val transr =
            TransR(numEntities, numEdges, DIMENSION).also {
                it.initialize(manager, DataType.FLOAT32, input.shape)
            }
        val model =
            Model.newInstance("transR").also {
                it.block = transr
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
        println(trainer.trainingResult)
        eTrainer.close()
        trainer.close()

        println(transr.getEdges())
        println(transr.getEntities())

        val test = manager.create(longArrayOf(1, 1, 2))
        println(predictor.predict(NDList(test)).singletonOrThrow())

        val result = ResultEvalTransR(inputList, manager.newSubManager(), predictor, numEntities, transR = transr)
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
        model.close()
    }
}

/** Marker class for TestTransR example. */
class TestTransR

package jp.live.ugai.tugraph

import ai.djl.Model
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.listener.EpochTrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.tracker.Tracker
import ai.djl.translate.NoopTranslator
import jp.live.ugai.tugraph.demo.buildTriplesInfo
import jp.live.ugai.tugraph.demo.newTrainer
import jp.live.ugai.tugraph.eval.ResultEvalQuatE
import jp.live.ugai.tugraph.train.DenseAdagrad
import jp.live.ugai.tugraph.train.EmbeddingTrainer
import jp.live.ugai.tugraph.train.HingeLossLoggingListener

/**
 * Runs a complete example: loads triples from data/sample.csv, builds and trains a QuatE embedding model,
 * performs a sample prediction, evaluates head/tail rankings, and prints results.
 *
 * The function covers data loading, model and trainer initialization, embedding training with EmbeddingTrainer,
 * prediction via a NoopTranslator-based predictor, evaluation using ResultEvalQuatE, and prints training metrics
 * and evaluation outputs. All created resources are closed before the function exits.
 */
fun main() {
    NDManager.newBaseManager().use { manager ->
        val csvReader = CsvToNdarray(manager)
        val input = csvReader.read("data/sample.csv")
        println(input)
        val triples = buildTriplesInfo(input)
        val quate =
            QuatE(triples.numEntities, triples.numEdges, DIMENSION).also {
                it.initialize(manager, DataType.FLOAT32, input.shape)
            }
        val model =
            Model.newInstance("quate").also {
                it.block = quate
            }

        val lrt = Tracker.fixed(ADAGRAD_LEARNING_RATE)
        val adagrad = DenseAdagrad.builder().optLearningRateTracker(lrt).build()

        val config =
            DefaultTrainingConfig(Loss.l1Loss()) // Placeholder loss; EmbeddingTrainer computes its own.
                .optOptimizer(adagrad) // Optimizer
                .optDevices(manager.engine.getDevices(1)) // single GPU
                .addTrainingListeners(EpochTrainingListener(), HingeLossLoggingListener()) // Hinge loss logging

        val trainer = newTrainer(model, config, Shape(BATCH_SIZE.toLong(), TRIPLE))

        val eTrainer = EmbeddingTrainer(manager.newSubManager(), input, triples.numEntities, trainer, NEPOCH)
        eTrainer.training()
        eTrainer.close()
        println(trainer.trainingResult)

        println(quate.getEdges())
        println(quate.getEntities())

        val predictor = model.newPredictor(NoopTranslator())
        val test = manager.create(longArrayOf(1, 1, 2))
        test.use {
            println(predictor.predict(NDList(it)).singletonOrThrow())
        }

        val result =
            ResultEvalQuatE(
                triples.inputList,
                manager.newSubManager(),
                predictor,
                triples.numEntities,
                quatE = quate,
                higherIsBetter = true,
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
        predictor.close()
        trainer.close()
        model.close()
    }
}

/** Marker class for TestQuatE example. */
class TestQuatE

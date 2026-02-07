package jp.live.ugai.tugraph

import ai.djl.Model
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.listener.EpochTrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Adam
import ai.djl.training.tracker.Tracker
import ai.djl.translate.NoopTranslator
import jp.live.ugai.tugraph.demo.buildTriplesInfo
import jp.live.ugai.tugraph.demo.newTrainer
import jp.live.ugai.tugraph.eval.ResultEvalHyperComplEx
import jp.live.ugai.tugraph.train.EmbeddingTrainer
import jp.live.ugai.tugraph.train.HingeLossLoggingListener

/**
 * Runs an end-to-end HyperComplEx demo that loads triples from CSV, trains embeddings, prints learned
 * model parameters and a sample prediction, evaluates head/tail ranking metrics, and closes all resources.
 *
 * The program:
 * - Loads triples from data/sample.csv and builds an in-memory list of (head, relation, tail) triples.
 * - Determines entity and relation counts and computes adaptive HyperComplEx dimensions.
 * - Initializes a HyperComplEx block, wraps it in a DJL Model, configures training (optimizer, listeners),
 *   and trains embeddings via EmbeddingTrainer.
 * - Prints training results, the internal HyperComplEx parameter tensors, and a sample prediction.
 * - Evaluates and prints tail and head ranking results using ResultEvalHyperComplEx.
 *
 * All created NDManager, model, trainer, predictor, and evaluator resources are closed before exit.
 */
fun main() {
    NDManager.newBaseManager().use { manager ->
        val csvReader = CsvToNdarray(manager)
        val input = csvReader.read("data/sample.csv")
        println(input)
        val triples = buildTriplesInfo(input)
        val dims = adaptiveHyperComplExDims(triples.numEntities, DIMENSION)
        val hyperComplEx =
            HyperComplEx(triples.numEntities, triples.numEdges, dims.dH, dims.dC, dims.dE).also {
                it.initialize(manager, DataType.FLOAT32, input.shape)
            }
        val model =
            Model.newInstance("hyper-complex").also {
                it.block = hyperComplEx
            }

        val lrt = Tracker.fixed(HYPERCOMPLEX_ADAM_LEARNING_RATE)
        val adam =
            Adam
                .builder()
                .optLearningRateTracker(lrt)
                .optBeta1(HYPERCOMPLEX_ADAM_BETA1)
                .optBeta2(HYPERCOMPLEX_ADAM_BETA2)
                .optEpsilon(HYPERCOMPLEX_ADAM_EPSILON)
                .optClipGrad(HYPERCOMPLEX_ADAM_CLIP_GRAD)
                .build()

        val config =
            DefaultTrainingConfig(Loss.l1Loss()) // Placeholder loss; EmbeddingTrainer computes its own.
                .optOptimizer(adam) // Optimizer
                .optDevices(manager.engine.getDevices(1)) // single GPU
                .addTrainingListeners(EpochTrainingListener(), HingeLossLoggingListener()) // Hinge loss logging

        val trainer = newTrainer(model, config, Shape(BATCH_SIZE.toLong(), TRIPLE))

        val eTrainer = EmbeddingTrainer(manager.newSubManager(), input, triples.numEntities, trainer, NEPOCH)
        eTrainer.training()
        eTrainer.close()
        println(trainer.trainingResult)

        hyperComplEx.getEdgesH().use { println(it) }
        hyperComplEx.getEdgesC().use { println(it) }
        hyperComplEx.getEdgesE().use { println(it) }
        hyperComplEx.getEntitiesH().use { println(it) }
        hyperComplEx.getEntitiesC().use { println(it) }
        hyperComplEx.getEntitiesE().use { println(it) }

        val predictor = model.newPredictor(NoopTranslator())
        val test = manager.create(longArrayOf(1, 1, 2))
        test.use {
            println(predictor.predict(NDList(it)).singletonOrThrow())
        }

        val result =
            ResultEvalHyperComplEx(
                triples.inputList,
                manager.newSubManager(),
                predictor,
                triples.numEntities,
                hyperComplEx = hyperComplEx,
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

/** Marker class for the HyperComplEx demo entry point. */
class TestHyperComplEx

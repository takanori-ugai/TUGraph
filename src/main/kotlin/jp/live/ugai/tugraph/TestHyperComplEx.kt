package jp.live.ugai.tugraph

import ai.djl.Model
import ai.djl.metric.Metrics
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

/**
 * Runs an end-to-end HyperComplEx demo.
 *
 * The demo loads triples from `data/sample.csv`, builds a model and trainer, runs embedding training,
 * prints learned parameters and a sample prediction, evaluates head/tail ranking metrics, and then
 * closes all resources.
 */
fun main() {
    NDManager.newBaseManager().use { manager ->
        val csvReader = CsvToNdarray(manager)
        val input = csvReader.read("data/sample.csv")
        println(input)
        val numOfTriples = input.shape[0]
        val flat = input.toLongArray()
        val inputList = ArrayList<LongArray>(numOfTriples.toInt())
        var idx = 0
        repeat(numOfTriples.toInt()) {
            inputList.add(longArrayOf(flat[idx], flat[idx + 1], flat[idx + 2]))
            idx += 3
        }
        val headMax =
            input.get(":, 0").use { col ->
                col.max().use { it.toLongArray()[0] }
            }
        val tailMax =
            input.get(":, 2").use { col ->
                col.max().use { it.toLongArray()[0] }
            }
        val relMax =
            input.get(":, 1").use { col ->
                col.max().use { it.toLongArray()[0] }
            }
        val numEntities = maxOf(headMax, tailMax) + 1
        val numEdges = relMax + 1
        val dims = adaptiveHyperComplExDims(numEntities, DIMENSION)
        val hyperComplEx =
            HyperComplEx(numEntities, numEdges, dims.dH, dims.dC, dims.dE).also {
                it.initialize(manager, DataType.FLOAT32, input.shape)
            }
        val model =
            Model.newInstance("hyper-complex").also {
                it.block = hyperComplEx
            }

        val lrt = Tracker.fixed(HYPERCOMPLEX_ADAM_LEARNING_RATE)
        val adam =
            Adam.builder()
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

        val trainer =
            model.newTrainer(config).also {
                it.initialize(Shape(BATCH_SIZE.toLong(), TRIPLE))
                it.metrics = Metrics()
            }

        val eTrainer = EmbeddingTrainer(manager.newSubManager(), input, numEntities, trainer, NEPOCH)
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
            ResultEval(
                inputList,
                manager.newSubManager(),
                predictor,
                numEntities,
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

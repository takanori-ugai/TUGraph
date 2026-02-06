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
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import ai.djl.translate.NoopTranslator
import jp.live.ugai.tugraph.eval.ResultEvalDistMult

/**
 * Trains and evaluates a DistMult knowledge-graph embedding model using triples loaded from "data/sample.csv" and prints training progress, model state, a sample prediction, and head/tail evaluation results.
 *
 * The program loads triples, derives entity and relation counts, initializes the DistMult model and trainer, runs embedding training, runs a sample prediction, evaluates results with ResultEvalDistMult, prints the tail and head metrics, and closes resources.
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
        val distMult =
            DistMult(numEntities, numEdges, DIMENSION).also {
                it.initialize(manager, DataType.FLOAT32, input.shape)
            }
        val model =
            Model.newInstance("distmult").also {
                it.block = distMult
            }

        val lrt = Tracker.fixed(LEARNING_RATE)
        val sgd = Optimizer.sgd().setLearningRateTracker(lrt).build()

        val config =
            DefaultTrainingConfig(Loss.l1Loss()) // Placeholder loss; EmbeddingTrainer computes its own.
                .optOptimizer(sgd) // Optimizer
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

        println(distMult.getEdges())
        println(distMult.getEntities())

        val predictor = model.newPredictor(NoopTranslator())
        val test = manager.create(longArrayOf(1, 1, 2))
        test.use {
            println(predictor.predict(NDList(it)).singletonOrThrow())
        }

        val result =
            ResultEvalDistMult(
                inputList,
                manager.newSubManager(),
                predictor,
                numEntities,
                distMult = distMult,
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

/** Marker class for TestDistMult example. */
class TestDistMult

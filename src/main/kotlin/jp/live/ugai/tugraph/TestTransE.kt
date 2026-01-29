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

/** Runs TransE training and evaluation on a TSV dataset. */
/**
 * Runs an end-to-end TransE training and evaluation pipeline using the TSV dataset at "data/ex/train2id-hrt.txt".
 *
 * Loads triples into memory, constructs and initializes a TransE model and DJL Trainer with an SGD optimizer,
 * performs embedding training for the configured number of epochs, evaluates head and tail predictions, and
 * prints progress and results to standard output. DJL resources (managers, model, trainer, predictor) are
 * created and closed within the function's scope.
 */
fun main() {
    NDManager.newBaseManager().use { manager ->
        val tsvReader = TsvToNdarray(manager)
        val input = tsvReader.read("data/ex/train2id-hrt.txt")
        println(input.get("0:10"))
        val numOfTriples = input.shape[0]
        val flat = input.toLongArray()
        val inputList = ArrayList<LongArray>(numOfTriples.toInt())
        var idx = 0
        repeat(numOfTriples.toInt()) {
            inputList.add(longArrayOf(flat[idx], flat[idx + 1], flat[idx + 2]))
            idx += 3
        }
        println("End Loading")
        val headMax = input.get(":, 0").max().toLongArray()[0]
        val tailMax = input.get(":, 2").max().toLongArray()[0]
        val relMax = input.get(":, 1").max().toLongArray()[0]
        val numEntities = maxOf(headMax, tailMax) + 1
        val numEdges = relMax + 1
        println("End Loading2")
        val transe =
            TransE(numEntities, numEdges, DIMENSION).also {
                it.initialize(manager, DataType.FLOAT16, input.shape)
            }
        println("End Loading3")
        val model =
            Model.newInstance("transe").also {
                it.block = transe
            }
        val predictor = model.newPredictor(NoopTranslator())

        val lrt = Tracker.fixed(LEARNING_RATE)
        val sgd = Optimizer.sgd().setLearningRateTracker(lrt).build()

        val config =
            DefaultTrainingConfig(Loss.l1Loss()) // Placeholder loss; EmbeddingTrainer computes its own.
                .optOptimizer(sgd) // Optimizer
                .optDevices(manager.engine.getDevices(1)) // single GPU
                .addTrainingListeners(EpochTrainingListener(), HingeLossLoggingListener()) // Hinge loss logging
        println("End Loading4")

        val trainer =
            model.newTrainer(config).also {
                it.initialize(Shape(BATCH_SIZE.toLong(), TRIPLE))
                it.metrics = Metrics()
            }

        println("End Loading5")
        val eTrainer = EmbeddingTrainer(manager.newSubManager(), input, numEntities, trainer, NEPOCH)
        println("Training Start")
        eTrainer.training()
        eTrainer.close()
        println("Training Result")
        println(trainer.trainingResult)

        println("Edge")
        println(transe.getEdges())
        println("Entities")
        println(transe.getEntities())

        val test = manager.create(longArrayOf(1, 1, 2))
        test.use {
            print("Predict (False):")
            println(predictor.predict(NDList(it)).singletonOrThrow().toFloatArray()[0])
        }

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
        model.close()
    }
}

/** Marker class for TestTransE example. */
class TestTransE
package jp.live.ugai.tugraph

import ai.djl.Model
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.listener.EpochTrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import ai.djl.translate.NoopTranslator
import jp.live.ugai.tugraph.demo.buildTriplesInfo
import jp.live.ugai.tugraph.demo.newTrainer
import jp.live.ugai.tugraph.eval.ResultEvalTransE
import jp.live.ugai.tugraph.train.EmbeddingTrainer
import jp.live.ugai.tugraph.train.HingeLossLoggingListener

/**
 * Runs the full TransE experiment: loads triples from CSV, constructs and initializes the model,
 * trains entity and relation embeddings, performs a sample prediction, evaluates head/tail rankings,
 * prints results, and releases resources.
 *
 * This function is the program entry point and orchestrates data loading, model/trainer setup,
 * training via EmbeddingTrainer, a sanity prediction, evaluation with ResultEvalTransE, and cleanup.
 */
fun main() {
    NDManager.newBaseManager().use { manager ->
        val csvReader = CsvToNdarray(manager)
        val input = csvReader.read("data/sample.csv")
        println(input)
        val triples = buildTriplesInfo(input)
        val transe =
            TransE(triples.numEntities, triples.numEdges, DIMENSION).also {
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

        val trainer = newTrainer(model, config, input.shape)

        val eTrainer = EmbeddingTrainer(manager.newSubManager(), input, triples.numEntities, trainer, NEPOCH)
        eTrainer.training()
        eTrainer.close()
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

        val result =
            ResultEvalTransE(
                triples.inputList,
                manager.newSubManager(),
                predictor,
                triples.numEntities,
                transE = transe,
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
        trainer.close()
        predictor.close()
        model.close()
    }
}

/** Marker class for Test6 example. */
class Test6

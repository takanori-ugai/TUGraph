package jp.live.ugai.tugraph

import ai.djl.Model
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.ParameterStore
import ai.djl.training.listener.EpochTrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.tracker.Tracker
import ai.djl.translate.NoopTranslator
import jp.live.ugai.tugraph.demo.buildTriplesInfo
import jp.live.ugai.tugraph.demo.newTrainer
import jp.live.ugai.tugraph.eval.ResultEvalRotatE
import jp.live.ugai.tugraph.train.DenseAdagrad
import jp.live.ugai.tugraph.train.EmbeddingTrainer
import jp.live.ugai.tugraph.train.HingeLossLoggingListener

/**
 * Runs a complete example that loads triple data, trains a RotatE embedding model, evaluates results, and computes Matryoshka scores.
 *
 * Loads triples from data/sample.csv, constructs and trains a RotatE model with an EmbeddingTrainer, computes nested-dimension
 * Matryoshka RotatE scores for a small batch of triples, performs a single prediction, evaluates head/tail metrics using ResultEvalRotatE,
 * and closes all allocated resources.
 */
fun main() {
    NDManager.newBaseManager().use { manager ->
        val csvReader = CsvToNdarray(manager)
        val input = csvReader.read("data/sample.csv")
        println(input)
        val triples = buildTriplesInfo(input)

        val rotate =
            RotatE(triples.numEntities, triples.numEdges, DIMENSION).also {
                it.initialize(manager, DataType.FLOAT32, input.shape)
            }
        val model =
            Model.newInstance("rotate-matryoshka").also {
                it.block = rotate
            }

        val lrt = Tracker.fixed(ADAGRAD_LEARNING_RATE)
        val adagrad = DenseAdagrad.builder().optLearningRateTracker(lrt).build()

        val config =
            DefaultTrainingConfig(Loss.l1Loss()) // Placeholder loss; EmbeddingTrainer computes its own.
                .optOptimizer(adagrad) // Optimizer
                .optDevices(manager.engine.getDevices(1)) // single GPU
                .addTrainingListeners(EpochTrainingListener(), HingeLossLoggingListener()) // Hinge loss logging

        val trainer = newTrainer(model, config, Shape(BATCH_SIZE.toLong(), TRIPLE))

        val eTrainer =
            EmbeddingTrainer(manager.newSubManager(), input, triples.numEntities, trainer, NEPOCH, enableMatryoshka = true)
        eTrainer.training()
        eTrainer.close()
        println(trainer.trainingResult)

        val predictor = model.newPredictor(NoopTranslator())

        // Matryoshka scores using nested dimensions on RotatE triples.
        val matryoshkaDims = MATRYOSHKA_ROTATE_DIMS
        val firstBatch = input.get("0:${minOf(4, triples.numTriples)}, :")
        firstBatch.use { batch ->
            manager.newSubManager().use { sm ->
                val parameterStore = ParameterStore(sm, true)
                val device = batch.device
                val entities = rotate.getEntities(parameterStore, device, false)
                val edges = rotate.getEdges(parameterStore, device, false)
                val embDim = entities.shape[1]
                val usable =
                    matryoshkaDims.filter { dim ->
                        dim > 0L && dim <= embDim && dim % 2L == 0L
                    }
                require(usable.isNotEmpty()) {
                    "No valid Matryoshka dims for RotatE (embDim=$embDim, dims=${matryoshkaDims.contentToString()})."
                }
                for (dim in usable) {
                    val score = rotate.score(batch, entities, edges, dim)
                    try {
                        println("Matryoshka RotatE score (totalDim=$dim): $score")
                    } finally {
                        score.close()
                    }
                }
            }
        }

        val test = manager.create(longArrayOf(1, 1, 2))
        test.use {
            println(predictor.predict(NDList(it)).singletonOrThrow())
        }

        val result =
            ResultEvalRotatE(
                triples.inputList,
                manager.newSubManager(),
                predictor,
                triples.numEntities,
                rotatE = rotate,
                higherIsBetter = false,
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

/** Marker class for TestMatryoshkaRotatE example. */
class TestMatryoshkaRotatE

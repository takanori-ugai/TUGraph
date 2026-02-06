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
import ai.djl.training.tracker.Tracker
import ai.djl.translate.NoopTranslator
import jp.live.ugai.tugraph.eval.ResultEvalQuatE
import jp.live.ugai.tugraph.train.DenseAdagrad
import jp.live.ugai.tugraph.train.EmbeddingTrainer
import jp.live.ugai.tugraph.train.HingeLossLoggingListener

/**
 * Runs an end-to-end example that trains, inspects, predicts with, and evaluates a QuatE embedding model using "data/sample.csv".
 *
 * Reads triples from the CSV, derives dataset sizes (entities/relations), initializes a QuatE block and a Model, configures and runs
 * an embedding training loop, computes Matryoshka dot scores for a small batch of triples, performs a single prediction, evaluates
 * head and tail prediction quality using ResultEvalQuatE, and prints training results, Matryoshka scores, a sample prediction, and
 * evaluation summaries. All created DJL/NDManager resources are closed before exit.
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

        val quate =
            QuatE(numEntities, numEdges, DIMENSION).also {
                it.initialize(manager, DataType.FLOAT32, input.shape)
            }
        val model =
            Model.newInstance("quate-matryoshka").also {
                it.block = quate
            }

        val lrt = Tracker.fixed(ADAGRAD_LEARNING_RATE)
        val adagrad = DenseAdagrad.builder().optLearningRateTracker(lrt).build()

        val config =
            DefaultTrainingConfig(Loss.l1Loss()) // Placeholder loss; EmbeddingTrainer computes its own.
                .optOptimizer(adagrad) // Optimizer
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

        val predictor = model.newPredictor(NoopTranslator())

        // Matryoshka scores using nested dimensions on entity embeddings.
        val matryoshkaDims = MATRYOSHKA_QUATE_DIMS
        val matryoshka = Matryoshka(matryoshkaDims)
        val firstBatch = input.get("0:${minOf(4, numOfTriples.toInt())}, :")
        firstBatch.use { batch ->
            val heads = batch.get(":, 0")
            val tails = batch.get(":, 2")
            try {
                val ent = quate.getEntities()
                val headEmb = ent.get(heads)
                val tailEmb = ent.get(tails)
                try {
                    val scores = matryoshka.dotScores(headEmb, tailEmb)
                    for (i in scores.indices) {
                        println("Matryoshka dot score (dim=${matryoshkaDims[i]}): ${scores[i]}")
                        scores[i].close()
                    }
                } finally {
                    headEmb.close()
                    tailEmb.close()
                }
            } finally {
                heads.close()
                tails.close()
            }
        }

        val test = manager.create(longArrayOf(1, 1, 2))
        test.use {
            println(predictor.predict(NDList(it)).singletonOrThrow())
        }

        val result =
            ResultEvalQuatE(
                inputList,
                manager.newSubManager(),
                predictor,
                numEntities,
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

/** Marker class for TestMatryoshka example. */
class TestMatryoshka

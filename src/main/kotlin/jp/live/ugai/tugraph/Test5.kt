package jp.live.ugai.tugraph

import ai.djl.Model
import ai.djl.metric.Metrics
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Parameter
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.EasyTrain
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.initializer.NormalInitializer
import ai.djl.training.listener.TrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import ai.djl.translate.NoopTranslator
import org.apache.commons.csv.CSVFormat
import java.io.FileInputStream
import java.io.InputStreamReader

/**
 * Runs a minimal TransE training and prediction example using the CSV file at `data/sample.csv`.
 *
 * Reads records from the CSV into tensors and an in-memory list, constructs an ArrayDataset,
 * initializes a TransE model, performs one epoch of training with L2 loss and SGD, and prints
 * parsed rows, membership checks, model edges/entities, and prediction outputs to standard output.
 *
 * This function performs I/O (reads `data/sample.csv`) and produces console output; it also
 * allocates and uses a DJL NDManager and model resources which are closed automatically.
 */
fun main() {
    NDManager.newBaseManager().use { manager ->
        var input = manager.zeros(Shape(0), DataType.INT64)

        val arr = mutableListOf<List<Long>>()
        FileInputStream("data/sample.csv").use { istream ->
            InputStreamReader(istream, "UTF-8").use { ireader ->
                val records = CSVFormat.EXCEL.parse(ireader)
                records.forEach {
                    val list = it.map { it.trim().toLong() }
                    input = input.concat(manager.create(list.toLongArray()))
                    println(list)
                    arr.add(list)
                }
            }
        }
        println(arr.contains(listOf<Long>(0, 0, 1)))
        input = input.reshape(arr.size.toLong(), input.size() / arr.size)

        val labels = manager.create(floatArrayOf(0f, 0f, 1f, 0f, 1f))
        val batchSize = 1
        val dataset =
            ArrayDataset
                .Builder()
                .setData(input) // set the features
                .optLabels(labels) // set the labels
                .setSampling(batchSize, true) // set the batch size and random sampling
                .build()

        val numEntities = 4L
        val numEdges = 2L
        val dim = 100L
        val transe = TransE(numEntities, numEdges, dim)
        transe.setInitializer(NormalInitializer(), Parameter.Type.WEIGHT)

        Model.newInstance("transe").use { model ->
            model.block = transe

            val l2loss = Loss.l2Loss()
            val lrt = Tracker.fixed(0.1f)
            val sgd = Optimizer.sgd().setLearningRateTracker(lrt).build()

            val config =
                DefaultTrainingConfig(l2loss)
                    .optOptimizer(sgd) // Optimizer
                    .optDevices(manager.engine.getDevices(1)) // single GPU
                    .apply { TrainingListener.Defaults.logging().forEach { addTrainingListeners(it) } } // Logging

            model.newPredictor(NoopTranslator()).use { predictor ->
                model.newTrainer(config).use { trainer ->
                    trainer.initialize(Shape(4, 3))
                    val metrics = Metrics()
                    trainer.metrics = metrics

                    val epochNum = 1
                    repeat(epochNum) {
                        for (batch in trainer.iterateDataset(dataset)) {
                            EasyTrain.trainBatch(trainer, batch)
                            trainer.step()
                            batch.close()
                        }
                    }

                    println(transe.getEdges())
                    println(transe.getEntities())
                    println(predictor.predict(NDList(input)).singletonOrThrow())
                    val test = manager.create(longArrayOf(1, 1, 2))
                    println(predictor.predict(NDList(test)).singletonOrThrow())
                }
            }
        }
    }
}

/** Marker class for Test5 example. */
class Test5

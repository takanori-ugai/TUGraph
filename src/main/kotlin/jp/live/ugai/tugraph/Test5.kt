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
import java.io.InputStream
import java.io.InputStreamReader
import kotlin.random.Random

/**
 * Runs a complete TransE training and inference workflow using data from `data/sample.csv`.
 *
 * Loads CSV records into an NDArray dataset, constructs labels and an ArrayDataset, initializes
 * and trains a TransE model for a fixed number of epochs, and prints model edges, entities,
 * and predictions for the training input and a sample test tensor.
 */
fun main() {
    NDManager.newBaseManager().use { manager ->
        var input = manager.zeros(Shape(0), DataType.INT64)

        val arr = mutableListOf<List<Long>>()
        var istream: InputStream = FileInputStream("data/sample.csv")
        var ireader = InputStreamReader(istream, "UTF-8")
        val records = CSVFormat.EXCEL.parse(ireader)
        records.forEach {
            val list = it.map { it.trim().toLong() }
            input = input.concat(manager.create(list.toLongArray()))
            println(list)
            arr.add(list)
        }
        println(arr.contains(listOf<Long>(0, 0, 1)))
        input = input.reshape(arr.size.toLong(), input.size() / arr.size)

        val labels = manager.create(floatArrayOf(0f, 0f, 1f, 0f, 1f))
        val batchSize = 1
        val dataset =
            ArrayDataset.Builder()
                .setData(input) // set the features
                .optLabels(labels) // set the labels
                .setSampling(batchSize, true) // set the batch size and random sampling
                .build()

        val transe = TransE(4, 2, 100)
        transe.setInitializer(NormalInitializer(), Parameter.Type.WEIGHT)
        transe.initialize(manager, DataType.FLOAT32, input.shape)

        val model = Model.newInstance("transe")
        model.block = transe

        val predictor = model.newPredictor(NoopTranslator())
//    println(predictor.predict(NDList(input)).singletonOrThrow())
//    println(transe.getEdges())
//    println(transe.getEntities())

        val l2loss = Loss.l2Loss()
        val lrt = Tracker.fixed(0.1f)
        val sgd = Optimizer.sgd().setLearningRateTracker(lrt).build()

        val config =
            DefaultTrainingConfig(l2loss)
                .optOptimizer(sgd) // Optimizer (loss function)
                .optDevices(manager.engine.getDevices(1)) // single GPU
                .apply { TrainingListener.Defaults.logging().forEach { addTrainingListeners(it) } } // Logging

        val trainer = model.newTrainer(config)
        trainer.initialize(Shape(4, 3))
        val metrics = Metrics()
        trainer.metrics = metrics

        val loss = mutableListOf<Float>()
        val epochNum = 1
        repeat(epochNum) {
            var l0 = 0f
            for (batch in trainer.iterateDataset(dataset)) {
                val x = batch.data.head()
                println(x.toLongArray()[2])
                val z = x.toLongArray().clone()
                val n = Random.nextLong(4)
                z[2] = n
                println("$n, ${z.toList()}, ${arr.contains(z.toList())}")
                val y = batch.labels.head()
                val f0 = trainer.forward(NDList(x))
                val l = f0.singletonOrThrow().sub(y).abs()
//                print(l)
                l0 += l.sum().toFloatArray()[0] / x.size(1)
                EasyTrain.trainBatch(trainer, batch)
                trainer.step()
                batch.close()
            }
//        trainer.notifyListeners { listener -> listener.onEpoch(trainer) }
            loss.add(l0)
//        transe.normalize()
        }

        println(transe.getEdges())
        println(transe.getEntities())
//    println(loss)
        println(predictor.predict(NDList(input)).singletonOrThrow())
        val test = manager.create(longArrayOf(1, 1, 2))
        println(predictor.predict(NDList(test)).singletonOrThrow())
    }
}

/** Marker class for Test5 example. */
class Test5

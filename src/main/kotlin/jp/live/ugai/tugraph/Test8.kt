package jp.live.ugai.tugraph

import ai.djl.Model
import ai.djl.basicdataset.tabular.CsvDataset
import ai.djl.metric.Metrics
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Parameter
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.initializer.NormalInitializer
import ai.djl.training.listener.TrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import ai.djl.translate.NoopTranslator
import org.apache.commons.csv.CSVFormat

fun main() {
    val manager = NDManager.newBaseManager()
    val csvReader = CsvToNdarray(manager)

    val input = csvReader.read("data/sample.csv")
    println(input)
    val numOfTriples = input.shape[0]
    val inputList = mutableListOf<List<Long>>()
    (0 until numOfTriples).forEach {
        inputList.add(input.get(it).toLongArray().toList())
    }

    val transe = TransE(NUM_ENTITIES, NUM_EDGES, DIMENSION).also {
        it.setInitializer(NormalInitializer(), Parameter.Type.WEIGHT)
        it.initialize(manager, DataType.FLOAT32, input.shape)
    }
    val model = Model.newInstance("transe").also {
        it.block = transe
    }
    val predictor = model.newPredictor(NoopTranslator())

    val lrt = Tracker.fixed(LEARNING_RATE)
    val sgd = Optimizer.sgd().setLearningRateTracker(lrt).build()

    val config = DefaultTrainingConfig(Loss.l1Loss())
        .optOptimizer(sgd) // Optimizer (loss function)
        .optDevices(manager.engine.getDevices(1)) // single GPU
        .addTrainingListeners(*TrainingListener.Defaults.logging()) // Logging

    val trainer = model.newTrainer(config).also {
        it.initialize(Shape(NUM_ENTITIES, TRIPLE))
        it.metrics = Metrics()
    }
    val csv = CsvDataset.CsvBuilder()
        .optCsvUrl("file:./data/sample.csv")
        .setCsvFormat(CSVFormat.Builder.create().setHeader("Col1", "Col2", "Col3").build())
        .addNumericFeature("Col1")
        .addNumericFeature("Col2")
        .addNumericFeature("Col3")
        .addNumericLabel("Col3")
        .setSampling(3, false)
        .build()
    csv.prepare()
    trainer.iterateDataset(csv).forEach {
        println(it.data)
        println(it.data[0])
    }
    val dataset = ArrayDataset.Builder()
        .setData(
            manager.arange(0, 4).reshape(1, 4),
            manager.arange(5, 9).reshape(1, 4),
            manager.arange(10, 14).reshape(1, 4),
            manager.arange(0, 4).reshape(1, 4),
            manager.arange(5, 9).reshape(1, 4),
            manager.arange(10, 14).reshape(1, 4)
        )
        .optLabels(
            manager.ones(Shape(1)),
            manager.zeros(Shape(1)),
            manager.zeros(Shape(1)),
            manager.ones(Shape(1)),
            manager.zeros(Shape(1)),
            manager.zeros(Shape(1))
        )
        .setSampling(2, true)
        .build()
    for (batch in trainer.iterateDataset(dataset)) {
        println(batch)
        println(batch.data[2])
        for (minibatch in batch.split(trainer.getDevices(), false)) {
            println(minibatch.data.size)
            println(minibatch.data[1])
        }
    }
    val ll = trainer.loss.evaluate(NDList(manager.create(floatArrayOf(1f))), NDList(manager.create(1.5f)))
    println(ll)
}

class Test8

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

/** Runs a TransE training example with training and validation datasets. */
fun main() {
    val manager = NDManager.newBaseManager()
    val input = manager.create(longArrayOf(2, 0, 1, 2, 1, 3, 0, 0, 1, 0, 1, 2), Shape(4, 3))
    val labels = manager.create(floatArrayOf(0f, 0f, 1f, 0f))
    val dataset =
        ArrayDataset.Builder()
            .setData(input) // set the features
            .optLabels(labels) // set the labels
            .setSampling(BATCH_SIZE, true) // set the batch size and random sampling
            .build()
    val validationSet =
        ArrayDataset.Builder()
            .setData(input) // set the features
            .optLabels(labels) // set the labels
            .setSampling(BATCH_SIZE, false) // set the batch size and random sampling
            .build()

    val transe = TransE(4, 2, DIMENSION)
    transe.setInitializer(NormalInitializer(), Parameter.Type.WEIGHT)
    transe.initialize(manager, DataType.FLOAT32, input.shape)

    val model = Model.newInstance("transe")
    model.block = transe

    val predictor = model.newPredictor(NoopTranslator())
//    println(predictor.predict(NDList(input)).singletonOrThrow())
//    println(transe.getEdges())
//    println(transe.getEntities())

    val l2loss = Loss.l2Loss()
    val lrt = Tracker.fixed(LEARNING_RATE)
    val sgd = Optimizer.sgd().setLearningRateTracker(lrt).build()

    val config =
        DefaultTrainingConfig(l2loss)
            .optOptimizer(sgd) // Optimizer (loss function)
            .optDevices(manager.engine.getDevices(1)) // single GPU
            .apply { TrainingListener.Defaults.logging().forEach { addTrainingListeners(it) } } // Logging

    val trainer = model.newTrainer(config)
//    trainer.initialize(input.shape)
    val metrics = Metrics()
    trainer.metrics = metrics

    EasyTrain.fit(trainer, NEPOCH, dataset, validationSet)
    println(trainer.trainingResult)
    println(trainer.metrics.metricNames)
    println(metrics.getMetric("train_epoch_L2Loss")[5].value)

    println(transe.getEdges())
    println(transe.getEntities())
//    println(loss)
    println(predictor.predict(NDList(input)).singletonOrThrow())
    val test = manager.create(longArrayOf(1, 1, 2))
    println(predictor.predict(NDList(test)).singletonOrThrow())
}

/** Marker class for Test3 example. */
class Test3

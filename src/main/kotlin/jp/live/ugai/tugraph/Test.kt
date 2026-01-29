package jp.live.ugai.tugraph

import ai.djl.Model
import ai.djl.metric.Metrics
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.listener.TrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker

/** Runs a basic TransE training smoke test. */
fun main() {
    val manager = NDManager.newBaseManager()
    val input = manager.create(longArrayOf(2, 0, 1, 2, 1, 3, 0, 0, 1, 0, 1, 2), Shape(4, 3))

    val transe = TransE(4, 2, 100)
    transe.initialize(manager, DataType.FLOAT32, input.shape)

    val model = Model.newInstance("transe")
    model.block = transe

    val lrt = Tracker.fixed(0.005f)
    val sgd = Optimizer.sgd().setLearningRateTracker(lrt).build()

    val config =
        DefaultTrainingConfig(Loss.l1Loss())
            .optOptimizer(sgd) // Optimizer (loss function)
            .optDevices(manager.engine.getDevices(1)) // single GPU
            .apply { TrainingListener.Defaults.logging().forEach { addTrainingListeners(it) } } // Logging

    val trainer = model.newTrainer(config)
    trainer.initialize(Shape(4, 3))
    val metrics = Metrics()
    trainer.metrics = metrics

    val eTrainer = EmbeddingTrainer(manager.newSubManager(), input, 4, trainer, 1000)
    eTrainer.training()

    println(transe.getEdges())
    println(transe.getEntities())
}

/** Marker class for Test example. */
class Test

package jp.live.ugai.tugraph

import ai.djl.Model
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

fun main() {
    val manager = NDManager.newBaseManager()
    val input = manager.create(longArrayOf(2, 0, 1, 2, 1, 3, 0, 0, 1, 0, 1, 2), Shape(4, 3))
    val labels = manager.create(floatArrayOf(0f, 0f, 1f, 0f))
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
    val lrt = Tracker.fixed(0.005f)
    val sgd = Optimizer.sgd().setLearningRateTracker(lrt).build()

    val config =
        DefaultTrainingConfig(l2loss)
            .optOptimizer(sgd) // Optimizer (loss function)
            .optDevices(manager.engine.getDevices(1)) // single GPU
            .addTrainingListeners(*TrainingListener.Defaults.logging()) // Logging

    val trainer = model.newTrainer(config)
    trainer.initialize(Shape(4, 3))
    val metrics = Metrics()
    trainer.metrics = metrics

    val loss = mutableListOf<Float>()
    val epochNum = 1000
    (0..epochNum).forEach {
        var l0 = 0f
        for (batch in trainer.iterateDataset(dataset)) {
            val x0 = batch.data.head()
            val y = batch.labels.head()
            trainer.newGradientCollector().use { gc ->
                val f0 = trainer.forward(NDList(x0))
                val l = f0.singletonOrThrow().pow(2).sqrt().sub(y).abs()
//                print(l)
                l0 += l.sum().toFloatArray()[0] / x0.size(1)
//                val l = point.pow(2).sum(intArrayOf(1)).sqrt().sub(y).abs()
//                l0 += l.sum().toFloatArray()[0]/ X.size(1)
//    val l : NDArray = l2loss.evaluate( NDList(NDArrays.pow(2, point).sum(intArrayOf(1)).sqrt()),
//    NDList(manager.create(floatArrayOf(0f,0f,1f,0f))))
//             if(i % 100 == 0) {
//                  println("l = $l")
//             }
                gc.backward(l)
                trainer.step()
//    println(model.block.getParameters().valueAt(0).getArray())
                gc.close()
            }
        }
        loss.add(l0)
    }

    println(transe.getEdges())
    println(transe.getEntities())
//    println(loss)
    println(predictor.predict(NDList(input)).singletonOrThrow())
    val test = manager.create(longArrayOf(1, 1, 2))
    println(predictor.predict(NDList(test)).singletonOrThrow())
    //   println(predictor.predict(NDList(test)).singletonOrThrow().pow(2).sum().sqrt())
}

class Test

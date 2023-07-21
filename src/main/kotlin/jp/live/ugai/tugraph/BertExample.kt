package jp.live.ugai.tugraph

import ai.djl.engine.Engine
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Activation
import ai.djl.nn.transformer.BertBlock
import ai.djl.nn.transformer.BertMaskedLanguageModelBlock
import ai.djl.nn.transformer.BertMaskedLanguageModelLoss
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.dataset.Dataset
import ai.djl.training.optimizer.Adam

fun main() {
    // Create an engine.
    val engine = Engine.getInstance()

    // Create a manager.
    val manager: NDManager = engine.newBaseManager()

    val builder = BertBlock.builder().setTokenDictionarySize(100).build()
    // Create a BERT masked language model block.
    val block = BertMaskedLanguageModelBlock(builder, Activation::relu)

    // Create a dataset.
    val dataset: Dataset = ArrayDataset.Builder()
        .setData(manager.randomInteger(100, 128, Shape(128), DataType.INT32)) // set the features
        .optLabels(manager.randomInteger(100, 128, Shape(128), DataType.INT32)) // set the labels
        .setSampling(3, true) // set the batch size and random sampling
        .build()

    // Create a training configuration.
    val loss = BertMaskedLanguageModelLoss(1, 2, 1)
    val config = DefaultTrainingConfig(loss).optOptimizer(Adam.adam().build())
//        config.setOptimizer(engine.getOptimizer("adam"))
//        config.setLearningRate(0.001)
//        config..setMetric(Accuracy())

    // Train the model.
    //    val result: TrainingResult = block.fit(dataset, config)

    // Print the training result.
    //    println(result)
}

class BertExample

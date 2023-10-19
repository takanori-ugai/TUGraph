package jp.live.ugai.tugraph

import ai.djl.Model
import ai.djl.modality.nlp.DefaultVocabulary
import ai.djl.modality.nlp.embedding.TrainableWordEmbedding
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Activation
import ai.djl.nn.SequentialBlock
import ai.djl.nn.transformer.TransformerEncoderBlock
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.ParameterStore
import ai.djl.training.loss.SoftmaxCrossEntropyLoss
import ai.djl.training.optimizer.Optimizer

/**
 * Main function to demonstrate the usage of various components in the code.
 */
fun main() {
    // Create NDManager
    val manager0 = NDManager.newBaseManager()

    // Create vocabulary and trainable word embedding
    val vos = DefaultVocabulary.builder().add(listOf("I", "am", "a", "dog")).build()
    val emb = TrainableWordEmbedding(vos, 4)
    emb.initialize(manager0, DataType.FLOAT32)

    // Forward pass through the trainable word embedding
    val ff = emb.forward(ParameterStore(), NDList(manager0.create(intArrayOf(0, 1), Shape(1, 2))), false)
    println(ff.head())
//    println(emb.embedWord(NDManager.newBaseManager().create(intArrayOf(0,1))))

    // Create softmax cross-entropy loss
    val loss = SoftmaxCrossEntropyLoss("SmCeLoss", 1.0F, -1, false, true)
    val config = DefaultTrainingConfig(loss)
        .optOptimizer(Optimizer.adam().build())
//        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT)

    // Create transformer encoder block and sequential block
    val block = TransformerEncoderBlock(4, 2, 10, 0.5F, Activation::softSign)
    val seq = SequentialBlock()
    seq.add(emb)
    seq.add(block)

    // Create model and set the sequential block as its block
    val model = Model.newInstance("model")
    model.setBlock(seq)

    val trainer = model.newTrainer(config)
    // the unused GradientCollector is for BatchNorm to know it is on training mode
//    val collector = trainer.newGradientCollector()

    // Initialize trainer and create input data and labels
    val inputShape = Shape(1, 2, 4)
    trainer.initialize(Shape(1, 2))
    val manager = trainer.getManager()
    val data = manager.create(floatArrayOf(1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F))
        .reshape(inputShape)
    val labels = manager.create(floatArrayOf(1.0F, -0.5F, 0.4F, 1.0F, 1.0F, 1.0F, -1.0F, -1.0F))
        .reshape(inputShape)

    // Create expected output for comparison
    val expected = manager.create(
        floatArrayOf(
            0.7615f,
            0.7615f,
            0.7615f,
            0.7615f,
            0.964f,
            0.964f,
            0.964f,
            0.964f
        ),
        Shape(1, 2, 4)
    )
    println(expected)

    // Training loop
    for (epoch in 0..100) {
        trainer.newGradientCollector().use { gc ->
            val result = trainer.forward(NDList(manager.create(intArrayOf(1, 2), Shape(1, 2))))
            println(result.head())
//            println(result.size)
            val lossValue = loss.evaluate(NDList(labels), NDList(result.head()))
            println(lossValue.getFloat())
            gc.backward(lossValue)
            gc.close()
        }
        trainer.step()
    }
}

/**
 * TestTransformer class for testing TransformerEncoder class.
 */
class TestTransformer

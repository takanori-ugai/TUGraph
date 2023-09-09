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

fun main() {
    val manager0 = NDManager.newBaseManager()
    val vos = DefaultVocabulary.builder().add(listOf("I", "am", "a", "dog")).build()
    val emb = TrainableWordEmbedding(vos, 4)
    emb.initialize(manager0, DataType.FLOAT32)
    val ff = emb.forward(ParameterStore(), NDList(manager0.create(intArrayOf(0, 1), Shape(1, 2))), false)
    println(ff.head())
//    println(emb.embedWord(NDManager.newBaseManager().create(intArrayOf(0,1))))
    val loss = SoftmaxCrossEntropyLoss("SmCeLoss", 1.0F, -1, false, true)
    val config = DefaultTrainingConfig(loss)
        .optOptimizer(Optimizer.adam().build())
//        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT)
    val block = TransformerEncoderBlock(4, 2, 10, 0.5F, Activation::softSign)
    val seq = SequentialBlock()
    seq.add(emb)
    seq.add(block)
    val model = Model.newInstance("model")
    model.setBlock(seq)

    val trainer = model.newTrainer(config)
    // the unused GradientCollector is for BatchNorm to know it is on training mode
//    val collector = trainer.newGradientCollector()
    val inputShape = Shape(1, 2, 4)
    trainer.initialize(Shape(1, 2))
    val manager = trainer.getManager()
    val data = manager.create(floatArrayOf(1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F))
        .reshape(inputShape)
    val labels = manager.create(floatArrayOf(1.0F, -0.5F, 0.4F, 1.0F, 1.0F, 1.0F, -1.0F, -1.0F))
        .reshape(inputShape)
    val expected = manager.create(
        floatArrayOf(
            00.7615f,
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
class TestTransformer

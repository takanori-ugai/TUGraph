package jp.live.ugai.tugraph

import ai.djl.Model
import ai.djl.metric.Metrics
import ai.djl.modality.nlp.DefaultVocabulary
import ai.djl.modality.nlp.embedding.TrainableWordEmbedding
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Activation
import ai.djl.nn.SequentialBlock
import ai.djl.nn.core.Linear
import ai.djl.nn.transformer.TransformerEncoderBlock
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.listener.TrainingListener
import ai.djl.training.loss.Loss
import ai.djl.translate.NoopTranslator

/** Example driver for training and evaluating a toy word embedding model. */
object EmbeddingExample {
    /** Runs the embedding example workflow. */
    @JvmStatic
    fun main(args: Array<String>) {
        val sentence = listOf("I", "am", "a", "dog", "am", "a", "<START>", "<END>")
        val dic =
            DefaultVocabulary
                .builder()
                .add(sentence)
                .optUnknownToken()
                .build()

        NDManager.newBaseManager().use { manager ->
            val sizeOfSentence = 8L
            val numOfSentence = 3L
            val sizeOfMatrix = sizeOfSentence * numOfSentence
            val numOfWords = dic.size() - 3 // <START>(4) <END>(5) <Unknown>(6)

            val lang = manager.randomInteger(0, numOfWords, Shape(sizeOfMatrix), DataType.INT32)
            println("LANG: $lang")
            val features =
                manager
                    .full(Shape(numOfSentence, 1), 4)
                    .concat(lang.reshape(numOfSentence, sizeOfSentence), 1)
            println("Features: $features")
            val lang2 =
                lang
                    .reshape(numOfSentence, sizeOfSentence)
                    .concat(manager.full(Shape(numOfSentence, 1), 5), 1)
                    .reshape((sizeOfSentence + 1) * numOfSentence)
            println(lang2)
            val labels =
                manager
                    .eye(numOfWords.toInt() + 2)
                    .get(lang2)
                    .reshape(numOfSentence, sizeOfSentence + 1, numOfWords + 2)
            println("Labels: $labels")
            val dataset = loadArray(features, labels, 2, true)
            println(dic.getIndex("<END>"))
            val emb = TrainableWordEmbedding(dic, 20)
            val net = SequentialBlock()
            val linearBlock =
                Linear
                    .builder()
                    .optBias(true)
                    .setUnits(numOfWords + 2)
                    .build()
            val trans = TransformerEncoderBlock(20, 2, 6, 0.5F, Activation::relu)
            net.add(emb)
            net.add(trans)
            net.add(linearBlock)
            val model = Model.newInstance("lin-reg")
            model.block = net
            val l = Loss.softmaxCrossEntropyLoss("SMCELOSS", 1.0F, -1, false, true)
            val config =
                DefaultTrainingConfig(l)
//            .optOptimizer(sgd) // Optimizer (loss function)
                    .apply { TrainingListener.Defaults.logging().forEach { addTrainingListeners(it) } } // Logging
            val trainer = model.newTrainer(config)
//        trainer.initialize(Shape(1, 2))
            val metrics = Metrics()
            trainer.metrics = metrics
            val numEpochs = 10
            val translator = NoopTranslator()
            val predictor = model.newPredictor(translator)
            for (epoch in 1..numEpochs) {
                println("Epoch $epoch")
                // Iterate over dataset
                var lossV = 0.0F
                for (batch in trainer.iterateDataset(dataset)) {
                    // Update loss and evaulator
                    trainer.newGradientCollector().use { collector ->
//                    println(batch.data[0])
//                    println(batch.data.size)
                        val preds = trainer.forward(batch.data)
//                    println(preds[0])
//                    println(batch.labels[0])
                        var time = System.nanoTime()
                        val lossValue = trainer.loss.evaluate(batch.labels, preds)
                        lossV += lossValue.getFloat()
//                    println("LossValue: $lossValue")
                        collector.backward(lossValue)
                        trainer.addMetric("backward", time)
                        time = System.nanoTime()
                        trainer.addMetric("training-metrics", time)
                    }

                    // Update parameters
                    trainer.step()
                    batch.close()
                }
                println("LOSS VALUE : $lossV")
                // reset training and validation evaluators at end of epoch
                trainer.notifyListeners { listener: TrainingListener ->
                    listener.onEpoch(
                        trainer,
                    )
                }
            }
            println("HELLO")
            val ppp = (predictor.predict(NDList(manager.create(intArrayOf(1, 0, 2), Shape(1, 3)))) as NDList)
            println(ppp[0].argMax(2))
            println(ppp[0])
        }
    }

    /**
     * Builds an ArrayDataset from feature and label NDArrays.
     *
     * @param features Input features.
     * @param labels Target labels.
     * @param batchSize Batch size for sampling.
     * @param shuffle Whether to shuffle the dataset.
     * @return Configured ArrayDataset instance.
     */
    fun loadArray(
        features: NDArray,
        labels: NDArray,
        batchSize: Int,
        shuffle: Boolean,
    ): ArrayDataset =
        ArrayDataset
            .Builder()
            .setData(features) // set the features
            .optLabels(labels) // set the labels
            .setSampling(batchSize, shuffle) // set the batch size and random sampling
            .build()
}

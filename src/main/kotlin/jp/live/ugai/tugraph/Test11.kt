package jp.live.ugai.tugraph

import ai.djl.Model
import ai.djl.ModelException
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.nn.Activation
import ai.djl.nn.transformer.TransformerEncoderBlock
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.EasyTrain
import ai.djl.training.TrainingConfig
import ai.djl.training.dataset.Batch
import ai.djl.training.dataset.Dataset
import ai.djl.training.dataset.Record
import ai.djl.training.evaluator.Accuracy
import ai.djl.training.listener.TrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Adam
import ai.djl.training.tracker.Tracker
import ai.djl.translate.Batchifier
import ai.djl.translate.TranslateException
import ai.djl.util.Progress
import java.io.IOException

/** Embedding size used by the toy transformer example. */
const val EMBEDDING_SIZE = 64

/** Output directory for saving the transformer model. */
const val MODEL_DIR = "build/models/transformer"

/** Trains a tiny transformer encoder on a toy dataset. */
fun main() {
    val logger = org.slf4j.LoggerFactory.getLogger("Test11")
    try {
        val dataset = prepareDataset()
        println("AAAA")
        val model = prepareModel()
        println("BBBB")
        val trainingConfig = prepareTrainingConfig()
        println("CCCC")
        val trainer = model.newTrainer(trainingConfig)
        println("DDDD")
        println("EEEE")
//        trainer.initialize(Shape(1L, EMBEDDING_SIZE.toLong()))
        println("FFFF")

        EasyTrain.fit(trainer, 10, dataset, dataset)
//        model.save(Paths.get(MODEL_DIR), "transformer")
    } catch (e: IOException) {
        logger.error("Failed to run transformer example.", e)
    } catch (e: ModelException) {
        logger.error("Failed to run transformer example.", e)
    } catch (e: TranslateException) {
        logger.error("Failed to run transformer example.", e)
    }
}

private fun prepareDataset(): Dataset {
    val sentences =
        arrayOf(
            "This is a positive sentence.",
            "I love this product.",
            "The movie was great.",
            "I dislike this book.",
            "The restaurant had terrible service.",
            "I'm happy with the results.",
        )
    val labels = longArrayOf(1L, 1L, 1L, 0L, 0L, 1L)

    return CustomDataset(sentences, labels)
}

private fun prepareModel(): Model {
    val hiddenSize = 64
    val numHeads = 2
    val numLayers = 2
    val dropoutRate = 0.1f

    val transformerEncoder =
        TransformerEncoderBlock(
            hiddenSize,
            numLayers,
            numHeads,
            dropoutRate,
            Activation::relu,
        )

    return Model.newInstance("transformer").apply {
        block = transformerEncoder
    }
}

private fun prepareTrainingConfig(): TrainingConfig {
    val learningRate = Tracker.fixed(0.001f)
    val loss = Loss.softmaxCrossEntropyLoss()
    val trainerBuilder =
        DefaultTrainingConfig(loss)
//        .setDevices(Device.getDevices(1))
            .addEvaluator(Accuracy())
            .optOptimizer(Adam.builder().optLearningRateTracker(learningRate).build())
            .apply { TrainingListener.Defaults.logging().forEach { addTrainingListeners(it) } }

    return trainerBuilder
}

private class CustomDataset(private val sentences: Array<String>, private val labels: LongArray) : Dataset {
    override fun prepare(progress: Progress?) {
        // No specific preparation required for this custom dataset
    }

    override fun getData(manager: NDManager): MutableIterable<Batch> {
        val records = mutableListOf<Record>()
        for (i in sentences.indices) {
            val sentence = sentences[i]
            val label = labels[i]
            val data = NDList(manager.create(sentence))
            val labelList = NDList(manager.create(label))
            val record = Record(data, labelList)
            records.add(record)
        }
        val batchSize = 3
        val numBatches = 6 / batchSize
        val batches = mutableListOf<Batch>()

        for (i in 0 until numBatches) {
            val startIndex = i * batchSize
            val endIndex = startIndex + batchSize
            val batchRecords = records.subList(startIndex, endIndex)
            val data = NDList()
            val labels = NDList()

            for (record in batchRecords) {
                data.add(record.data[0])
                labels.add(record.labels[0])
            }

            val batch = Batch(manager, data, labels, 0, Batchifier.STACK, Batchifier.STACK, 0, 0)
            batches.add(batch)
        }

        return batches
    }
}

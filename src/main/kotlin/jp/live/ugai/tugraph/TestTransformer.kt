package jp.live.ugai.tugraph

import ai.djl.Model
import ai.djl.modality.nlp.DefaultVocabulary
import ai.djl.modality.nlp.embedding.TrainableWordEmbedding
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.nn.Activation
import ai.djl.nn.core.Linear
import ai.djl.nn.transformer.TransformerEncoderBlock
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.ParameterStore
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.loss.SoftmaxCrossEntropyLoss
import ai.djl.training.optimizer.Optimizer
import ai.djl.util.PairList

/**
 * Runs a self-contained example that trains and evaluates a tiny causal transformer language model.
 *
 * Trains a toy model on two short token sequences using DJL utilities, prints epoch loss every 20 epochs,
 * and prints final token-level accuracy after training.
 */
fun main() {
    NDManager.newBaseManager().use { manager ->

        // Tiny LM vocabulary
        val vocab =
            DefaultVocabulary
                .builder()
                .add(listOf("<BOS>", "<EOS>", "i", "am", "a", "dog", "cat"))
                .optUnknownToken()
                .build()

        // Next-token prediction (causal LM)
        val loss = SoftmaxCrossEntropyLoss("SmCeLoss", 1.0F, -1, true, true)
        val config =
            DefaultTrainingConfig(loss)
                .optOptimizer(Optimizer.adam().build())

        val block = CausalTransformerLMBlock(vocab, 8, 2, 16)
        val model = Model.newInstance("toy-transformer")
        model.setBlock(block)

        val trainer = model.newTrainer(config)
        trainer.initialize(Shape(1, 4))
        val vocabSize = vocab.size()

        val sequences =
            listOf(
                listOf("<BOS>", "i", "am", "a", "dog", "<EOS>"),
                listOf("<BOS>", "i", "am", "a", "cat", "<EOS>"),
            )
        val seqLen = 4
        val inputs = mutableListOf<IntArray>()
        val targets = mutableListOf<IntArray>()
        for (sentence in sequences) {
            val ids = sentence.map { vocab.getIndex(it).toInt() }
            for (i in 0..(ids.size - seqLen - 1)) {
                inputs.add(ids.subList(i, i + seqLen).toIntArray())
                targets.add(ids.subList(i + 1, i + seqLen + 1).toIntArray())
            }
        }
        val dataArr = IntArray(inputs.size * seqLen)
        val labelArr = IntArray(targets.size * seqLen)
        inputs.forEachIndexed { row, arr ->
            System.arraycopy(arr, 0, dataArr, row * seqLen, seqLen)
        }
        targets.forEachIndexed { row, arr ->
            System.arraycopy(arr, 0, labelArr, row * seqLen, seqLen)
        }
        val data = manager.create(dataArr, Shape(inputs.size.toLong(), seqLen.toLong()))
        val labels = manager.create(labelArr, Shape(targets.size.toLong(), seqLen.toLong()))

        val dataset =
            ArrayDataset
                .Builder()
                .setData(data)
                .optLabels(labels)
                .setSampling(2, true)
                .build()

        for (epoch in 1..100) {
            var epochLoss = 0f
            for (batch in trainer.iterateDataset(dataset)) {
                val x = batch.data.head()
                val y = batch.labels.head()
                trainer.newGradientCollector().use { gc ->
                    val logits = trainer.forward(NDList(x)).head()
                    val logits2 = logits.reshape(Shape(-1, vocabSize))
                    val y2 = y.reshape(Shape(-1))
                    val lossValue = loss.evaluate(NDList(y2), NDList(logits2))
                    epochLoss += lossValue.getFloat()
                    gc.backward(lossValue)
                }
                trainer.step()
            }
            if (epoch % 20 == 0) {
                println("Epoch $epoch loss=$epochLoss")
            }
        }

        var correct = 0
        var total = 0
        for (batch in trainer.iterateDataset(dataset)) {
            val x = batch.data.head()
            val y = batch.labels.head()
            val logits = trainer.forward(NDList(x)).head()
            val pred = logits.argMax(2)
            val yArr = y.toIntArray()
            val pArr = pred.toType(DataType.INT32, false).toIntArray()
            for (i in yArr.indices) {
                if (yArr[i] == pArr[i]) {
                    correct++
                }
                total++
            }
        }
        println("Accuracy: ${correct.toFloat() / total}")
    }
}

/** A minimal causal language model block built from DJL transformer components. */
class CausalTransformerLMBlock(
    private val vocab: DefaultVocabulary,
    private val embedDim: Int,
    private val numHeads: Int,
    private val ffnDim: Int,
) : AbstractBlock() {
    private val embedding: TrainableWordEmbedding
    private val encoder: TransformerEncoderBlock
    private val projection: Linear

    init {
        embedding = addChildBlock("embedding", TrainableWordEmbedding(vocab, embedDim)) as TrainableWordEmbedding
        val encoderBlock = TransformerEncoderBlock(embedDim, numHeads, ffnDim, 0.1F, Activation::relu)
        encoder = addChildBlock("encoder", encoderBlock) as TransformerEncoderBlock
        projection = addChildBlock("projection", Linear.builder().setUnits(vocab.size()).build()) as Linear
    }

    /**
     * Runs the forward pass with a causal attention mask.
     *
     * @param parameterStore Parameter store for the block.
     * @param inputs Input token IDs.
     * @param training Whether the block is in training mode.
     * @param params Additional parameters.
     * @return Output logits NDList.
     */
    override fun forwardInternal(
        parameterStore: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?,
    ): NDList {
        val tokens = inputs.head()
        val embedded = embedding.forward(parameterStore, NDList(tokens), training).head()
        val shape = tokens.shape
        val batch = shape.get(0)
        val seqLen = shape.get(1)
        val mask = causalMask(tokens.manager, batch, seqLen)
        val encoded = encoder.forward(parameterStore, NDList(embedded, mask), training).head()
        val logits = projection.forward(parameterStore, NDList(encoded), training).head()
        return NDList(logits)
    }

    /**
     * Computes output shapes for the given input shapes.
     *
     * @param inputShapes Input shapes for the block.
     * @return Output shapes for the block.
     */
    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
        val batch = inputShapes[0].get(0)
        val seqLen = inputShapes[0].get(1)
        return arrayOf(Shape(batch, seqLen, vocab.size()))
    }

    /**
     * Initializes child blocks with derived embedding shapes.
     *
     * @param manager NDManager used for initialization.
     * @param dataType Data type for parameters.
     * @param inputShapes Input shapes for initialization.
     */
    override fun initializeChildBlocks(
        manager: NDManager,
        dataType: DataType,
        vararg inputShapes: Shape,
    ) {
        embedding.initialize(manager, dataType, *inputShapes)
        val embedShape = Shape(inputShapes[0].get(0), inputShapes[0].get(1), embedDim.toLong())
        encoder.initialize(manager, dataType, embedShape)
        projection.initialize(manager, dataType, embedShape)
    }

    private fun causalMask(
        manager: NDManager,
        batch: Long,
        seqLen: Long,
    ): NDArray {
        val idx = manager.arange(seqLen.toInt())
        val i = idx.reshape(Shape(seqLen, 1))
        val j = idx.reshape(Shape(1, seqLen))
        val mask = i.gte(j).toType(DataType.FLOAT32, false)
        return mask.expandDims(0).repeat(0, batch)
    }
}

/** Marker class for TestTransformer example. */
class TestTransformer

package jp.live.ugai.tugraph

import ai.djl.modality.nlp.DefaultVocabulary
import ai.djl.modality.nlp.embedding.TrainableWordEmbedding
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.training.ParameterStore

/**
 * Demonstrates creating and using a TrainableWordEmbedding with a small vocabulary and prints example embeddings
 * and encodings.
 *
 * Sets up an NDManager and a ParameterStore, constructs a vocabulary and a trainable embedding, initializes it,
 * and runs several embed/encode/forward operations whose results are printed to standard output.
 */
/**
 * Demonstrates creating and using a TrainableWordEmbedding with a small vocabulary and prints various embeddings and outputs.
 *
 * Creates a base NDManager and ParameterStore, builds a DefaultVocabulary from ["2","1","4","3"], constructs and
 * initializes a TrainableWordEmbedding (4 embeddings, size 10), then exercises and prints:
 * - token embedding via embed(String)
 * - encoded token bytes via encode(String)
 * - embedding for an input array via embed(manager, Array<String>)
 * - forward pass outputs for index inputs and a sequence of ND operations (transpose, elementwise multiply, sum)
 * - the underlying parameter array(s)
 *
 * The NDManager is closed automatically via use to ensure proper resource cleanup.
 */
fun main() {
    NDManager.newBaseManager().use { manager ->
        val ps = ParameterStore(manager, false)
        val voList = listOf("2", "1", "4", "3")
        val voc = DefaultVocabulary(voList)
        val a =
            TrainableWordEmbedding
                .builder()
                .setVocabulary(voc)
                .optNumEmbeddings(4)
                .setEmbeddingSize(10)
                .build()
        a.initialize(manager, DataType.FLOAT32)
//    a.prepare(arrayOf(Shape(1,10)))
        println(a.embed("4"))
        println(String(a.encode("1")))
        println(a.embed(manager, arrayOf("1"))[0])
        println(a.forward(ps, NDList(manager.create(intArrayOf(0, 1))), false)[0])
        println(
            a
                .forward(ps, NDList(manager.create(intArrayOf(0, 1))), false)[0]
                .transpose()
                .muli(manager.create(intArrayOf(1, -1)))
                .sum(intArrayOf(1)),
        )
        println(a.parameters[0].value.array)
        println(a.parameters[0].value.array)
//    println(a.embedWord(manager, 1))
    }
}

/** Marker class for Test10 example. */
class Test10
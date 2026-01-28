package jp.live.ugai.tugraph

import ai.djl.modality.nlp.DefaultVocabulary
import ai.djl.modality.nlp.embedding.TrainableWordEmbedding
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.training.ParameterStore

/** Runs a quick embedding lookup demonstration. */
fun main() {
    val manager = NDManager.newBaseManager()
    val ps = ParameterStore(manager, false)
    val arr = manager.arange(10)
    val voList = listOf("2", "1", "4", "3")
    val voc = DefaultVocabulary(voList)
    val a = TrainableWordEmbedding.builder().setVocabulary(voc).optNumEmbeddings(4).setEmbeddingSize(10).build()
    a.initialize(manager, DataType.FLOAT32)
//    a.prepare(arrayOf(Shape(1,10)))
    println(a.embed("4"))
    println(String(a.encode("1")))
    println(a.embed(manager, arrayOf("1"))[0])
    println(a.forward(ps, NDList(manager.create(intArrayOf(0, 1))), false)[0])
    println(
        a.forward(ps, NDList(manager.create(intArrayOf(0, 1))), false)[0].transpose()
            .muli(manager.create(intArrayOf(1, -1))).sum(intArrayOf(1)),
    )
    println(a.parameters[0].value.array)
    println(a.parameters[0].value.array)
//    println(a.embedWord(manager, 1))
}

/** Marker class for Test10 example. */
class Test10

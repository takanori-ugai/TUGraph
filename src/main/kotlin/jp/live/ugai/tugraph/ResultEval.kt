package jp.live.ugai.tugraph

import ai.djl.inference.Predictor
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import javax.xml.crypto.Data

class ResultEval(val inputList: List<List<Long>>, val manager: NDManager, val predictor: Predictor<NDList, NDList>) {

    fun getTailResult(): Map<String, Float> {
        val res = mutableMapOf<String, Float>()
        val rankList = mutableListOf<Int>()
        inputList.forEach { triple ->
            val rank = predictor.predict(
                NDList(
                    manager.create(longArrayOf(triple[0], triple[1]))
                        .reshape(1, 2)
                        .repeat(0, NUM_ENTITIES)
                        .concat(manager.create(LongArray(NUM_ENTITIES.toInt()){it.toLong()}).reshape(NUM_ENTITIES, 1), 1)
                )
            ).singletonOrThrow().toFloatArray()
            val lengthOrder = rank.mapIndexed { index, fl -> Pair(index, fl) }
                .toList().sortedBy { it.second }
            rankList.add(lengthOrder.indexOf(Pair(triple[2].toInt(), rank[triple[2].toInt()])) + 1)
        }

        res["HIT@1"] = rankList.filter { it <= 1 }.size.toFloat() / rankList.size.toFloat()
        res["HIT@10"] = rankList.filter { it <= 10 }.size.toFloat() / rankList.size.toFloat()
        res["HIT@100"] = rankList.filter { it <= 100 }.size.toFloat() / rankList.size.toFloat()
        res["MRR"] = rankList.map { 1 / it.toFloat() }.sum() / rankList.size
        return res
    }

    fun getHeadResult(): Map<String, Float> {
        val res = mutableMapOf<String, Float>()
        val rankList = mutableListOf<Int>()
        inputList.forEach { triple ->
            val rank = mutableMapOf<Long, Float>()
            (0 until NUM_ENTITIES).forEach {
                val t0 = manager.create(longArrayOf(it, triple[1], triple[2]))
                rank[it] = predictor.predict(NDList(t0)).singletonOrThrow().toFloatArray()[0]
            }
            val lengthOrder = rank.toList().sortedBy { it.second }
            rankList.add(lengthOrder.indexOf(Pair(triple[0], rank[triple[0]])) + 1)
        }

        res["HIT@1"] = rankList.filter { it <= 1 }.size.toFloat() / rankList.size.toFloat()
        res["HIT@10"] = rankList.filter { it <= 10 }.size.toFloat() / rankList.size.toFloat()
        res["HIT@100"] = rankList.filter { it <= 100 }.size.toFloat() / rankList.size.toFloat()
        res["MRR"] = rankList.map { 1 / it.toFloat() }.sum() / rankList.size
        return res
    }

    fun close() {
        manager.close()
    }
}

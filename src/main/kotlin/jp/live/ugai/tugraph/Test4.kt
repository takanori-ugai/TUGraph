package jp.live.ugai.tugraph

import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import com.opencsv.CSVReader
import java.io.FileInputStream
import java.io.InputStream
import java.io.InputStreamReader

fun main() {
    val manager = NDManager.newBaseManager()
    var input = manager.zeros(Shape(0))

    val arr = mutableListOf<List<Long>>()
    var istream: InputStream = FileInputStream("data/sample.csv")
    var ireader: InputStreamReader = InputStreamReader(istream, "UTF-8")
    var reader: CSVReader = CSVReader(ireader)

    reader.forEach {
        val list = it.map { it.trim().toLong() }
        input = input.concat(manager.create(list.toLongArray()))
        println(list)
        arr.add(list)
    }
    println(arr.contains(listOf<Long>(0, 0, 1)))
    println(input.reshape(arr.size.toLong(), input.size() / arr.size))
}

class Test4

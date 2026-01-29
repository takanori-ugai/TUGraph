package jp.live.ugai.tugraph

import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import org.apache.commons.csv.CSVFormat
import java.io.FileInputStream
import java.io.InputStream
import java.io.InputStreamReader

/**
 * Application entry point that loads numeric CSV data and prints processing results.
 *
 * Reads "data/sample.csv" using UTF-8 and Excel CSV format, parses each record into a
 * list of Long values, appends rows into a DJL NDArray, and prints each parsed row.
 * After reading all records it prints whether the sequence [0, 0, 1] is present and
 * prints the NDArray reshaped to (rowCount, columnCount).
 */
fun main() {
    NDManager.newBaseManager().use { manager ->
        var input = manager.zeros(Shape(0), DataType.INT64)

        val arr = mutableListOf<List<Long>>()
        var istream: InputStream = FileInputStream("data/sample.csv")
        var ireader: InputStreamReader = InputStreamReader(istream, "UTF-8")
        val records = CSVFormat.EXCEL.parse(ireader)
        records.forEach {
            val list = it.map { it.trim().toLong() }
            input = input.concat(manager.create(list.toLongArray()))
            println(list)
            arr.add(list)
        }
        println(arr.contains(listOf<Long>(0, 0, 1)))
        println(input.reshape(arr.size.toLong(), input.size() / arr.size))
    }
}

/** Marker class for Test4 example. */
class Test4

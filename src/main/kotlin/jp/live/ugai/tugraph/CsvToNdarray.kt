package jp.live.ugai.tugraph

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import com.opencsv.CSVParserBuilder
import com.opencsv.CSVReaderBuilder
import java.io.FileInputStream
import java.io.InputStreamReader

class CsvToNdarray(val manager: NDManager) {

    fun read(path: String): NDArray {
        var input = manager.zeros(Shape(0), DataType.INT64)
        var istream = FileInputStream(path)
        var ireader = InputStreamReader(istream, "UTF-8")
//        var reader = CSVReader('\t', ireader)
        var reader = CSVReaderBuilder(ireader)
            .withCSVParser(
                CSVParserBuilder()
//                .withSeparator('\t')
                    .withSeparator(',')
                    .build()
            ).build()
        reader.forEach {
            val list = it.map { it.trim().toLong() }.toList()
//            val list0 = listOf(list.elementAt(0), list.elementAt(2), list.elementAt(1))
//            input = input.concat(manager.create(list0.toLongArray()))
            input = input.concat(manager.create(list.toLongArray()))
        }
        return input.reshape(input.size() / TRIPLE, TRIPLE)
    }
}

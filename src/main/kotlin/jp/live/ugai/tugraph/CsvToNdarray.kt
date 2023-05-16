package jp.live.ugai.tugraph

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import org.apache.commons.csv.CSVFormat
import java.io.FileInputStream
import java.io.InputStreamReader

class CsvToNdarray(val manager: NDManager) {

    fun read(path: String): NDArray {
        FileInputStream(path).use { istream ->
            InputStreamReader(istream, "UTF-8").use { ireader ->
//        val records = CSVFormat.EXCEL.parse(ireader)
                CSVFormat.TDF.parse(ireader).use { parser ->
                    val array = mutableListOf<Long>()
                    parser.records.forEach { record ->
                        array.addAll(record.map { value -> value.trim().toLong() })
                    }
                    return manager.create(array.toLongArray(), Shape(array.size / TRIPLE, TRIPLE))
                }
            }
        }
    }
}

package jp.live.ugai.tugraph

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import org.apache.commons.csv.CSVFormat
import java.io.FileInputStream
import java.io.InputStreamReader

/**
 * A utility class for reading CSV files and converting them to NDArrays.
 *
 * @property manager The NDManager instance used for creating NDArrays.
 */
class CsvToNdarray(val manager: NDManager) {
    /**
     * Reads a CSV file and converts it to an NDArray.
     *
     * @param path The path to the CSV file.
     * @return The NDArray containing the data from the CSV file.
     */
    fun read(path: String): NDArray {
        FileInputStream(path).use { istream ->
            InputStreamReader(istream, "UTF-8").use { ireader ->
//        val records = CSVFormat.EXCEL.parse(ireader) TDF
                CSVFormat.EXCEL.parse(ireader).use { parser ->
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

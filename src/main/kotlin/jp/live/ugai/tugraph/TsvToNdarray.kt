package jp.live.ugai.tugraph

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.InputStreamReader

/**
 * A utility class for reading TSV/whitespace files and converting them to NDArrays.
 *
 * @property manager The NDManager instance used for creating NDArrays.
 */
class TsvToNdarray(
    /** NDManager used to create NDArrays. */
    val manager: NDManager,
) {
    /**
     * Reads a TSV/whitespace file and converts it to an NDArray.
     *
     * This supports files like `data/ex/train2id-hrt.txt` where each row is:
     * head<TAB>relation<TAB>tail (or whitespace-separated). If the first line
     * contains a single integer count, it is skipped by default.
     *
     * @param path The path to the TSV file.
     * @param skipCountHeader Skip first line if it contains a single value.
     * @return The NDArray containing the data from the file.
     */
    fun read(path: String, skipCountHeader: Boolean = true): NDArray {
        FileInputStream(path).use { istream ->
            InputStreamReader(istream, Charsets.UTF_8).use { ireader ->
                BufferedReader(ireader).use { reader ->
                    val values = ArrayList<Long>()
                    var isFirst = true
                    reader.lineSequence().forEach { line ->
                        val trimmed = line.trim()
                        if (trimmed.isEmpty()) {
                            return@forEach
                        }
                        val parts = trimmed.split(Regex("\\s+"))
                        if (isFirst && skipCountHeader && parts.size == 1) {
                            isFirst = false
                            return@forEach
                        }
                        isFirst = false
                        parts.forEach { token ->
                            values.add(token.toLong())
                        }
                    }
                    require(values.size % TRIPLE.toInt() == 0) {
                        "Input file does not contain a multiple of 3 values (size=${values.size})."
                    }
                    return manager.create(values.toLongArray(), Shape(values.size / TRIPLE, TRIPLE))
                }
            }
        }
    }
}

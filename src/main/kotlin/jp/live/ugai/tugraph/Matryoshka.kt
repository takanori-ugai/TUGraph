package jp.live.ugai.tugraph

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.index.NDIndex

/**
 * Utility for Matryoshka (nested) embeddings.
 *
 * Matryoshka embeddings allow using the first k dimensions of a full embedding
 * as a valid, lower-capacity representation.
 */
class Matryoshka(
    /** Ordered list of nested dimensions to use (e.g., [32, 64, 128]). */
    val dims: LongArray,
) {
    init {
        require(dims.isNotEmpty()) { "dims must not be empty." }
        for (i in 1 until dims.size) {
            require(dims[i] > dims[i - 1]) { "dims must be strictly increasing." }
        }
    }

    /**
     * Create a view containing the first prefix dimensions of an embedding.
     *
     * @param embedding NDArray with shape (N, D) where D is the embedding width.
     * @param dim Prefix dimension to keep; must be greater than 0 and less than or equal to D.
     * @return An NDArray view with shape (N, dim) containing the first `dim` features for each item.
     */
    fun slice(
        embedding: NDArray,
        dim: Long,
    ): NDArray {
        require(dim > 0) { "dim must be > 0." }
        require(dim <= embedding.shape[1]) {
            "dim must be <= embedding dimension (${embedding.shape[1]}), was $dim."
        }
        return embedding.get(NDIndex(":, 0:$dim"))
    }

    /**
     * Computes dot-product scores for each configured dimension.
     *
     * @param left NDArray of shape (N, D).
     * @param right NDArray of shape (N, D).
     * @return List of NDArray scores, one per configured dimension, each of shape (N).
     */
    fun dotScores(
        left: NDArray,
        right: NDArray,
    ): List<NDArray> {
        require(left.shape == right.shape) { "left/right must have the same shape." }
        val maxDim = left.shape[1]
        val outputs = ArrayList<NDArray>(dims.size)
        for (dim in dims) {
            require(dim <= maxDim) { "dim $dim exceeds embedding dimension $maxDim." }
            val l = left.get(NDIndex(":, 0:$dim"))
            val r = right.get(NDIndex(":, 0:$dim"))
            val product = l.mul(r)
            val score = product.sum(intArrayOf(1))
            product.close()
            outputs.add(score)
            l.close()
            r.close()
        }
        return outputs
    }
}

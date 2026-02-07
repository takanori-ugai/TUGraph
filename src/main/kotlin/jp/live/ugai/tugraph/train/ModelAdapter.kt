package jp.live.ugai.tugraph.train

import jp.live.ugai.tugraph.HyperComplEx
import jp.live.ugai.tugraph.TransE
import jp.live.ugai.tugraph.TransR

/**
 * Adapter that exposes model-specific hooks needed during training.
 *
 * Use [from] to wrap a block and get the appropriate post-update behavior.
 */
internal sealed interface ModelAdapter {
    val block: Any

    /**
     * Executes any post-update normalization/projection required by the wrapped block.
     */
    fun postUpdateHook() {
        val current = block
        when (current) {
            is TransE -> current.normalize()
            is TransR -> current.normalize()
            is HyperComplEx -> current.projectHyperbolic()
            else -> Unit // Other blocks do not require post-update normalization.
        }
    }

    companion object {
        /**
         * Creates an adapter for the provided block.
         */
        fun from(block: Any): ModelAdapter =
            when (block) {
                is HyperComplEx -> HyperComplExAdapter(block)
                else -> DefaultAdapter(block)
            }
    }
}

/**
 * Adapter for HyperComplEx to expose the concrete block type.
 *
 * @property block HyperComplEx instance used during training.
 */
internal class HyperComplExAdapter(
    override val block: HyperComplEx,
) : ModelAdapter

/**
 * Default adapter for blocks that do not need extra hooks.
 *
 * @property block Underlying model block.
 */
internal class DefaultAdapter(
    override val block: Any,
) : ModelAdapter

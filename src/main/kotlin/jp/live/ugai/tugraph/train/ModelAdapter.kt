package jp.live.ugai.tugraph.train

import jp.live.ugai.tugraph.HyperComplEx
import jp.live.ugai.tugraph.TransE
import jp.live.ugai.tugraph.TransR

internal sealed interface ModelAdapter {
    val block: Any

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
        fun from(block: Any): ModelAdapter =
            when (block) {
                is HyperComplEx -> HyperComplExAdapter(block)
                else -> DefaultAdapter(block)
            }
    }
}

internal class HyperComplExAdapter(
    override val block: HyperComplEx,
) : ModelAdapter

internal class DefaultAdapter(
    override val block: Any,
) : ModelAdapter

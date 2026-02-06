package jp.live.ugai.tugraph.train

import jp.live.ugai.tugraph.HyperComplEx
import jp.live.ugai.tugraph.TransE
import jp.live.ugai.tugraph.TransR

internal sealed interface ModelAdapter {
    val block: Any
    val isHyperComplEx: Boolean

    fun postUpdateHook() {
        val current = block
        when (current) {
            is TransE -> current.normalize()
            is TransR -> current.normalize()
            is HyperComplEx -> current.projectHyperbolic()
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
) : ModelAdapter {
    override val isHyperComplEx: Boolean = true
}

internal class DefaultAdapter(
    override val block: Any,
) : ModelAdapter {
    override val isHyperComplEx: Boolean = false
}

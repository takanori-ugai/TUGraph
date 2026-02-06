package jp.live.ugai.tugraph.train

internal data class TripleKey(
    val head: Long,
    val rel: Long,
    val tail: Long,
)

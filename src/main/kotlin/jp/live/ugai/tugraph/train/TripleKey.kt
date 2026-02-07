package jp.live.ugai.tugraph.train

/**
 * Value object for fast membership checks of (head, relation, tail) triples.
 *
 * @property head Head entity ID.
 * @property rel Relation ID.
 * @property tail Tail entity ID.
 */
internal data class TripleKey(
    val head: Long,
    val rel: Long,
    val tail: Long,
)

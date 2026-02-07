package jp.live.ugai.tugraph

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.index.NDIndex

internal data class QuatView(
    val r: NDArray,
    val i: NDArray,
    val j: NDArray,
    val k: NDArray,
)

/**
 * Resolve Matryoshka dimensions for QuatE, returning component-level dimensions.
 *
 * When any dimension exceeds the per-component size, dimensions are interpreted as total embedding sizes
 * (multiples of 4) and converted to component sizes.
 */
internal fun resolveQuatEComponentDims(
    dims: LongArray,
    fullDim: Long,
    embDim: Long,
): List<Long> {
    val resolved = ArrayList<Long>(dims.size)
    val useTotalDims = dims.any { it > fullDim }
    for (d in dims) {
        val compDim = toComponentDim(d, fullDim, embDim, useTotalDims)
        if (compDim != null && compDim <= fullDim) {
            resolved.add(compDim)
        }
    }
    require(resolved.isNotEmpty()) { "No valid Matryoshka dims for QuatE (fullDim=$fullDim, embDim=$embDim)." }
    return resolved
}

/**
 * Resolve Matryoshka dimensions for QuatE and attach weights for each usable component dimension.
 */
internal fun resolveQuatEComponentDimsWithWeights(
    dims: LongArray,
    weights: FloatArray,
    fullDim: Long,
    embDim: Long,
): List<Pair<Long, Float>> {
    require(dims.size == weights.size) { "Matryoshka dims and weights must have the same length." }
    val componentDims = resolveQuatEComponentDims(dims, fullDim, embDim).toSet()
    val useTotalDims = dims.any { it > fullDim }
    val paired = ArrayList<Pair<Long, Float>>(dims.size)
    for (i in dims.indices) {
        val d = dims[i]
        val compDim = toComponentDim(d, fullDim, embDim, useTotalDims)
        if (compDim != null && componentDims.contains(compDim)) {
            paired.add(compDim to weights[i])
        }
    }
    return paired
}

private fun toComponentDim(
    dim: Long,
    fullDim: Long,
    embDim: Long,
    useTotalDims: Boolean,
): Long? =
    if (dim <= 0L) {
        null
    } else if (useTotalDims) {
        if (dim <= embDim && dim % 4L == 0L) dim / 4L else null
    } else {
        if (dim <= fullDim) dim else null
    }

/**
 * Compute Matryoshka QuatE scores for a batch of triples at the given component dimension.
 */
internal fun matryoshkaQuatEScore(
    input: NDArray,
    entities: NDArray,
    edges: NDArray,
    componentDim: Long,
    fullDim: Long,
): NDArray {
    require(componentDim > 0L) { "componentDim must be > 0." }
    require(componentDim <= fullDim) { "componentDim must be <= fullDim ($fullDim)." }
    val numTriples = input.size() / TRIPLE
    val parent = input.manager
    return parent.newSubManager().use { sm ->
        val triples = input.reshape(numTriples, TRIPLE).also { it.attach(sm) }
        val headIds = triples.get(NDIndex(":, 0")).also { it.attach(sm) }
        val relationIds = triples.get(NDIndex(":, 1")).also { it.attach(sm) }
        val tailIds = triples.get(NDIndex(":, 2")).also { it.attach(sm) }

        val heads = entities.get(headIds).also { it.attach(sm) }
        val relations = edges.get(relationIds).also { it.attach(sm) }
        val tails = entities.get(tailIds).also { it.attach(sm) }

        val h = splitQuaternion(heads, componentDim, fullDim, attachTo = { it.attach(sm) })
        val r = splitQuaternion(relations, componentDim, fullDim, attachTo = { it.attach(sm) })
        val t = splitQuaternion(tails, componentDim, fullDim, attachTo = { it.attach(sm) })

        // Hamilton product r ⊗ t
        val rtR =
            r.r
                .mul(t.r)
                .sub(r.i.mul(t.i))
                .sub(r.j.mul(t.j))
                .sub(r.k.mul(t.k))
                .also { it.attach(sm) }
        val rtI =
            r.r
                .mul(t.i)
                .add(r.i.mul(t.r))
                .add(r.j.mul(t.k))
                .sub(r.k.mul(t.j))
                .also { it.attach(sm) }
        val rtJ =
            r.r
                .mul(t.j)
                .sub(r.i.mul(t.k))
                .add(r.j.mul(t.r))
                .add(r.k.mul(t.i))
                .also { it.attach(sm) }
        val rtK =
            r.r
                .mul(t.k)
                .add(r.i.mul(t.j))
                .sub(r.j.mul(t.i))
                .add(r.k.mul(t.r))
                .also { it.attach(sm) }

        val score =
            h.r
                .mul(rtR)
                .add(h.i.mul(rtI))
                .add(h.j.mul(rtJ))
                .add(h.k.mul(rtK))
                .sum(intArrayOf(1))
                .also { it.attach(parent) }
        score
    }
}

internal fun splitQuaternion(
    base: NDArray,
    componentDim: Long,
    fullDim: Long,
    attachTo: (NDArray) -> Unit,
): QuatView =
    QuatView(
        base.get(NDIndex(":, 0:$componentDim")).also(attachTo),
        base.get(NDIndex(":, $fullDim:${fullDim + componentDim}")).also(attachTo),
        base.get(NDIndex(":, ${2 * fullDim}:${2 * fullDim + componentDim}")).also(attachTo),
        base.get(NDIndex(":, ${3 * fullDim}:${3 * fullDim + componentDim}")).also(attachTo),
    )

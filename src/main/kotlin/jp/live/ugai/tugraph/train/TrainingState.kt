package jp.live.ugai.tugraph.train

internal class TrainingState(
    var lastValidate: Float = Float.NaN,
    var firstNanReported: Boolean = false,
)

package jp.live.ugai.tugraph.train

/**
 * Mutable per-run state tracked across epochs.
 *
 * @property lastValidate Last computed validation loss.
 * @property firstNanReported Whether a NaN warning has been logged.
 */
internal class TrainingState(
    var lastValidate: Float = Float.NaN,
    var firstNanReported: Boolean = false,
)

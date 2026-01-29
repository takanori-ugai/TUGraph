package jp.live.ugai.tugraph

// Hyper Parameters

/** Default learning rate used in training examples. */
const val LEARNING_RATE = 0.005f

/** Default embedding dimension for example models. */
const val DIMENSION = 20L

/** Default number of epochs for example training runs. */
const val NEPOCH = 300

/** Default batch size for example datasets. */
const val BATCH_SIZE = 3

/** Max attempts to resample a negative before falling back. */
const val NEGATIVE_RESAMPLE_CAP = 10

/** Number of values in a knowledge graph triple. */
const val TRIPLE = 3L

// const val NUM_ENTITIES = 202487L
// const val NUM_ENTITIES = 7658L
// const val NUM_EDGES = 37L
// const val NUM_EDGES = 57L

/** Placeholder class to keep common constants in one file. */
class Commons

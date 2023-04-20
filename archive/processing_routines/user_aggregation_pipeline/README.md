# User Aggregation Pipeline

This code runs a pipeline which downloads the event history from MongoDB, summarizes the events by user
calculates tSNE embeddings, flags fraudulent users.

The pipeline can be run with `bash run_pipeline.sh`

Note that this pipeline pulls every event from MongoDB and takes almost an hour to run so just beware.
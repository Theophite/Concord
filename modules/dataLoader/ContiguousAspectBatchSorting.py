from mgds.pipelineModules.AspectBatchSorting import AspectBatchSorting


class ContiguousAspectBatchSorting(AspectBatchSorting):
    """Drop-in AspectBatchSorting that keeps each resolution bucket as a CONTIGUOUS run
    of batches across the epoch, instead of globally shuffling batches across buckets.

    Why: with CUDA-graph capture + Concord packed layers, interleaving image shapes (the
    base class's `rand.shuffle(batches)`) forces a graph release/recapture on every shape
    change AND makes the CUDA caching allocator churn differently-sized activation /
    graph-pool blocks -> fragmentation -> OOM on nominally-free memory. Keeping each bucket
    contiguous gives one stable allocation footprint per block; the shape (hence the graph
    and the Concord v_hat) changes only a handful of times per epoch.

    Still fully randomized per epoch (variation): the sample->batch assignment within a
    bucket, the batch order within a bucket, and the ORDER of the buckets are all reshuffled.
    Every bucket is present each epoch (blocks-within-epoch) -- the gentle variant, not
    whole-epoch-per-shape -- so it avoids single-shape forgetting.

    Only start() is overridden (get_item / length / IO are inherited). The bucket build and
    ordering are reimplemented rather than reaching into the base's name-mangled privates;
    no MGDS edit. Instantiated identically to AspectBatchSorting.
    """

    def start(self, variation: int):
        self._build_buckets()
        self.index_list = self._contiguous_order()

    def _build_buckets(self):
        # resolution -> [sample indices]  (mirrors the base __sort_resolutions)
        self.bucket_dict = {}
        for index in range(self._get_previous_length(self.resolution_in_name)):
            resolution = self._get_previous_item(self.current_variation, self.resolution_in_name, index)
            self.bucket_dict.setdefault(resolution, []).append(index)

    def _contiguous_order(self) -> list[int]:
        rand = self._get_rand(self.current_variation)
        bucket_dict = {key: value.copy() for (key, value) in self.bucket_dict.items()}

        # shuffle samples within each bucket, drop the partial-batch remainder
        for samples in bucket_dict.values():
            rand.shuffle(samples)
            for _ in range(len(samples) % self.batch_size):
                samples.pop()

        # random bucket ORDER, random batch order WITHIN each bucket, blocks kept contiguous
        bucket_keys = list(bucket_dict.keys())
        rand.shuffle(bucket_keys)

        index_list = []
        for bucket_key in bucket_keys:
            samples = bucket_dict[bucket_key]
            batch_order = list(range(len(samples) // self.batch_size))
            rand.shuffle(batch_order)
            for b in batch_order:
                index_list.extend(samples[b * self.batch_size:(b + 1) * self.batch_size])
        return index_list

void QuotaManager::EvictOriginData(
    const GURL& origin,
    StorageType type,
    EvictOriginDataCallback* callback) {
  DCHECK(io_thread_->BelongsToCurrentThread());
  DCHECK_EQ(type, kStorageTypeTemporary);

  eviction_context_.evicted_origin = origin;
  eviction_context_.evicted_type = type;
  eviction_context_.evict_origin_data_callback.reset(callback);

  DeleteOriginData(origin, type, callback_factory_.NewCallback(
      &QuotaManager::DidOriginDataEvicted));
}
